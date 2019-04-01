# coding=utf-8
#!/usr/bin/env python3
import datetime
import torch
from pprint import pformat
import models
from dataset import create_dataloader_train_cv
import fire
import losses
import logging
import pandas as pd
import kaldi_io
import yaml
import os
import numpy as np
from dcase_util.data import ManyHotEncoder, ProbabilityEncoder
from sklearn import metrics
import tableprint as tp
import sklearn.preprocessing as pre
import torchnet as tnt
import sed_eval
from torch._six import container_abcs
from itertools import repeat


class AUCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = []
        self.targets = []

    def add(self, outputs, targets):
        outputs, targets = np.atleast_2d(
            outputs.cpu().numpy(), targets.cpu().numpy())
        self.outputs.append(outputs)
        self.targets.append(targets)

    def value(self):
        return metrics.roc_auc_score(
            np.concatenate(self.targets, axis=0),
            np.concatenate(self.outputs, axis=0),
            average='macro')


class BinarySimilarMeter(object):
    """Only counts ones, does not consider zeros as being correct"""

    def __init__(self, sigmoid_output=False):
        super(BinarySimilarMeter, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.reset()

    def reset(self):
        self.correct = 0
        self.n = 0

    def add(self, output, target):
        if self.sigmoid_output:
            output = torch.sigmoid(output)
        output = output.round()
        self.correct += np.sum(np.logical_and(output, target).numpy())
        self.n += (target == 1).nonzero().shape[0]

    def value(self):
        if self.n == 0:
            return 0
        return (self.correct / self.n) * 100.


class BinaryAccuracyMeter(object):
    """Counts all outputs, including zero"""

    def __init__(self, sigmoid_output=False):
        super(BinaryAccuracyMeter, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.reset()

    def reset(self):
        self.correct = 0
        self.n = 0

    def add(self, output, target):
        if self.sigmoid_output:
            output = torch.sigmoid(output)
        output = output.round()
        self.correct += int((output == target).sum())
        self.n += np.prod(output.shape)

    def value(self):
        if self.n == 0:
            return 0
        return (self.correct / self.n) * 100.


def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --center ark:- ark:- |"
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr


def runepoch(dataloader, model, criterion, optimizer=None, dotrain=True, poolfun=lambda x, d: x.mean(d)):
    model = model.train() if dotrain else model.eval()
    # By default use average pooling
    utt_loss_meter = tnt.meter.AverageValueMeter()
    utt_acc_meter = BinarySimilarMeter()
    auc_meter = AUCMeter()
    with torch.set_grad_enabled(dotrain):
        for i, (features, utt_targets) in enumerate(dataloader):
            features = features.float().to(device)
            # Might be a bit taxing on the GPU to put all 500 * 10 labels there
            utt_targets = utt_targets.float().cpu()
            outputs = torch.sigmoid(model(features)).cpu()
            pooled_prob = poolfun(outputs, 1)
            loss = criterion(pooled_prob, utt_targets)
            utt_loss_meter.add(loss.item())
            auc_meter.add(pooled_prob.data, utt_targets.data)
            utt_acc_meter.add(pooled_prob.data, utt_targets.data)
            if dotrain:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return utt_loss_meter.value(), utt_acc_meter.value(), auc_meter.value()


def genlogger(outdir, fname):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read)
    # passed kwargs will override yaml config
    for key in kwargs.keys():
        assert key in yaml_config, "Parameter {} invalid!".format(key)
    return dict(yaml_config, **kwargs)


def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve

    :mode: can be ether 'loss' or 'acc'
    :returns: function that can be called, function returns true if criterion improved

    """
    assert mode in ('loss', 'acc')
    best_value = np.inf if mode == 'loss' else 0

    def comparator(x, best_x):
        return x < best_x if mode == 'loss' else x > best_x

    def inner(x):
        # rebind parent scope variable
        nonlocal best_value
        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)


def main(config='config/ReLU/0Pool/crnn_maxpool.yaml', **kwargs):
    """Trains a model on the given features and vocab.

    :features: str: Input features. Needs to be kaldi formatted file
    :config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
    :returns: None
    """

    config_parameters = parse_config_or_kwargs(config, **kwargs)
    outputdir = os.path.join(
        config_parameters['outputpath'],
        config_parameters['model'],
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%f'))
    try:
        os.makedirs(outputdir)
    except IOError:
        pass
    logger = genlogger(outputdir, 'train.log')
    logger.info("Storing data at: {}".format(outputdir))
    logger.info("<== Passed Arguments ==>")
    # Print arguments into logs
    for line in pformat(config_parameters).split('\n'):
        logger.info(line)

    kaldi_string = parsecopyfeats(
        config_parameters['features'], **config_parameters['feature_args'])

    scaler = getattr(
        pre, config_parameters['scaler'])(
        **config_parameters['scaler_args'])
    inputdim = -1
    logger.info(
        "<== Estimating Scaler ({}) ==>".format(
            scaler.__class__.__name__))
    for kid, feat in kaldi_io.read_mat_ark(kaldi_string):
        scaler.partial_fit(feat)
        inputdim = feat.shape[-1]
    assert inputdim > 0, "Reading inputstream failed"
    logger.info(
        "Features: {} Input dimension: {}".format(
            config_parameters['features'],
            inputdim))
    logger.info("<== Labels ==>")
    label_df = pd.read_csv(config_parameters['labels'], sep='\t')
    label_df.event_labels = label_df.event_labels.str.split(',')
    label_df = label_df.set_index('filename')
    uniquelabels = list(np.unique(
        [item
         for row in label_df.event_labels.values
         for item in row]))
    many_hot_encoder = ManyHotEncoder(
        label_list=uniquelabels,
        time_resolution=1
    )
    label_df['manyhot'] = label_df['event_labels'].apply(
        lambda x: many_hot_encoder.encode(x, 1).data.flatten())

    utt_labels = label_df.loc[:, 'manyhot'].to_dict()

    train_dataloader, cv_dataloader = create_dataloader_train_cv(
        kaldi_string,
        utt_labels,
        transform=scaler.transform,
        **config_parameters['dataloader_args'])
    model = getattr(
        models,
        config_parameters['model'])(
        inputdim=inputdim,
        output_size=len(uniquelabels),
        **config_parameters['model_args'])
    logger.info("<== Model ==>")
    for line in pformat(model).split('\n'):
        logger.info(line)
    optimizer = getattr(
        torch.optim, config_parameters['optimizer'])(
        model.parameters(),
        **config_parameters['optimizer_args'])

    scheduler = getattr(
        torch.optim.lr_scheduler,
        config_parameters['scheduler'])(
        optimizer,
        **config_parameters['scheduler_args'])
    criterion = getattr(losses, config_parameters['loss'])(
        **config_parameters['loss_args'])

    trainedmodelpath = os.path.join(outputdir, 'model.th')

    model = model.to(device)
    criterion_improved = criterion_improver(
        config_parameters['improvecriterion'])
    header = [
        'Epoch',
        'UttLoss(T)',
        'UttLoss(CV)',
        "UttAcc(T)",
        "UttAcc(CV)",
        "mAUC(CV)"]
    for line in tp.header(
        header,
            style='grid').split('\n'):
        logger.info(line)

    poolingfunction_name = config_parameters['poolingfunction']
    pooling_function = parse_poolingfunction(poolingfunction_name)
    for epoch in range(1, config_parameters['epochs']+1):
        train_utt_loss_mean_std, train_utt_acc, train_auc_utt = runepoch(
            train_dataloader, model, criterion, optimizer, dotrain=True, poolfun=pooling_function)
        cv_utt_loss_mean_std, cv_utt_acc, cv_auc_utt = runepoch(
            cv_dataloader, model,  criterion, dotrain=False, poolfun=pooling_function)
        logger.info(
            tp.row(
                (epoch,) +
                (train_utt_loss_mean_std[0],
                 cv_utt_loss_mean_std[0],
                 train_utt_acc, cv_utt_acc, cv_auc_utt),
                style='grid'))
        epoch_meanloss = cv_utt_loss_mean_std[0]
        if epoch % config_parameters['saveinterval'] == 0:
            torch.save({'model': model,
                        'scaler': scaler,
                        'encoder': many_hot_encoder,
                        'config': config_parameters},
                       os.path.join(outputdir, 'model_{}.th'.format(epoch)))
        # ReduceOnPlateau needs a value to work
        schedarg = epoch_meanloss if scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None
        scheduler.step(schedarg)
        if criterion_improved(epoch_meanloss):
            torch.save({'model': model,
                        'scaler': scaler,
                        'encoder': many_hot_encoder,
                        'config': config_parameters},
                       trainedmodelpath)
        if optimizer.param_groups[0]['lr'] < 1e-7:
            break
    logger.info(tp.bottom(len(header), style='grid'))
    logger.info("Results are in: {}".format(outputdir))
    return outputdir


def parse_poolingfunction(poolingfunction_name='mean'):
    if poolingfunction_name == 'mean':
        def pooling_function(x, d): return x.mean(d)
    elif poolingfunction_name == 'max':
        def pooling_function(x, d): return x.max(d)[0]
    elif poolingfunction_name == 'linear':
        def pooling_function(x, d): return (x**2).sum(d) / x.sum(d)
    elif poolingfunction_name == 'exp':
        def pooling_function(x, d): return (
            x.exp() * x).sum(d) / x.exp().sum(d)
    return pooling_function


def evaluate_threshold(
        model_path: str, features: str = "features/logmel_64/test.ark",
        result_filename='dev.txt',
        test_labels:
        str = "metadata/test/test.csv",
        threshold=0.5,
        window=1,
        hop_size=0.02):
    from dcase_util.data import ProbabilityEncoder, DecisionEncoder, ManyHotEncoder
    from dcase_util.containers import MetaDataContainer
    from scipy.signal import medfilt
    modeldump = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    model = modeldump['model']
    config_parameters = modeldump['config']
    scaler = modeldump['scaler']
    many_hot_encoder = modeldump['encoder']
    model_dirname = os.path.dirname(model_path)
    meta_container_resultfile = os.path.join(
        model_dirname, "pred_nowindow.txt")
    metacontainer = MetaDataContainer(filename=meta_container_resultfile)

    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    model = model.to(device).eval()

    probability_encoder = ProbabilityEncoder()
    decision_encoder = DecisionEncoder(
        label_list=many_hot_encoder.label_list
    )
    binarization_type = 'global_threshold' if isinstance(
        threshold, float) else 'class_threshold'
    # If class thresholds are given, then use those
    if isinstance(threshold, str):
        threshold = torch.load(threshold)
    windows = {k: window for k in many_hot_encoder.label_list}
    if isinstance(window, str):
        windows = torch.load(window)

    with torch.no_grad():
        for k, feat in kaldi_io.read_mat_ark(kaldi_string):
            # Add batch dim
            feat = torch.from_numpy(
                scaler.transform(feat)).to(device).unsqueeze(0)
            feat = model(feat)
            probabilities = torch.sigmoid(feat).cpu().numpy().squeeze(0)
            frame_decisions = probability_encoder.binarization(
                probabilities=probabilities,
                binarization_type=binarization_type,
                threshold=threshold,
                time_axis=0,
            )
            for i, label in enumerate(many_hot_encoder.label_list):
                label_frame_decisions = medfilt(
                    frame_decisions[:, i], kernel_size=windows[label])
                # Found only zeros, no activity, go on
                if (label_frame_decisions == 0).all():
                    continue
                estimated_events = decision_encoder.find_contiguous_regions(
                    activity_array=label_frame_decisions
                )
                for [onset, offset] in estimated_events:
                    metacontainer.append({'event_label': label,
                                          'onset': onset * hop_size,
                                          'offset': offset * hop_size,
                                          'filename': os.path.basename(k)
                                          })
    metacontainer.save()
    estimated_event_list = MetaDataContainer().load(
        filename=meta_container_resultfile)
    reference_event_list = MetaDataContainer().load(filename=test_labels)

    event_based_metric = event_based_evaluation(
        reference_event_list, estimated_event_list)
    onset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, offset=False)
    offset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, onset=False)
    onset_offset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list)
    # Utt wise Accuracy
    precision_labels = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, onset=False, offset=False, label=True)

    print(event_based_metric.__str__())
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("UttLabel", *precision_labels))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("Onset", *onset_scores))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("Offset", *offset_scores))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("On-Offset", *onset_offset_scores))

    result_filename = os.path.join(model_dirname, result_filename)

    with open(result_filename, 'w') as wp:
        wp.write(event_based_metric.__str__())
        wp.write('\n')
        wp.write("{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format(
            "UttLabel", *precision_labels))
        wp.write(
            "{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format("Onset", *onset_scores))
        wp.write(
            "{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format("Offset", *offset_scores))
        wp.write("{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format(
            "On-Offset", *onset_offset_scores))


def event_based_evaluation(reference_event_list, estimated_event_list):
    """ Calculate sed_eval event based metric for challenge

        Parameters
        ----------

        reference_event_list : MetaDataContainer, list of referenced events

        estimated_event_list : MetaDataContainer, list of estimated events

        Return
        ------

        event_based_metric : EventBasedMetrics

        """

    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        # evaluate_onset = False,
        # evaluate_offset = False,
        t_collar=0.200,
        percentage_of_length=0.2,
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return event_based_metric


def precision_recall_fscore_on_offset(reference_event_list, estimated_event_list, onset=True, offset=True, label=False):
    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))
    overall = {'ntp': 0, 'nsys': 0, 'nref': 0}

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        ntp, nsys, nref = _precision_recall_fscore_on_offset(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
            onset=onset,
            offset=offset,
            label=label
        )
        overall['ntp'] += ntp
        overall['nsys'] += nsys
        overall['nref'] += nref

    precision = sed_eval.metric.precision(
        Ntp=overall['ntp'], Nsys=overall['nsys'])
    recall = sed_eval.metric.recall(Ntp=overall['ntp'], Nref=overall['nref'])
    f_score = sed_eval.metric.f_measure(precision, recall)
    return precision, recall, f_score


def _precision_recall_fscore_on_offset(reference_event_list, estimated_event_list, onset=True, offset=True, label=False):
    # Evaluate only valid events
    import dcase_util
    valid_reference_event_list = dcase_util.containers.MetaDataContainer()
    for item in reference_event_list:
        if 'event_onset' in item and 'event_offset' in item and 'event_label' in item:
            valid_reference_event_list.append(item)

        elif 'onset' in item and 'offset' in item and 'event_label' in item:
            valid_reference_event_list.append(item)

    reference_event_list = valid_reference_event_list

    valid_estimated_event_list = dcase_util.containers.MetaDataContainer()
    for item in estimated_event_list:
        if 'event_onset' in item and 'event_offset' in item and 'event_label' in item:
            valid_estimated_event_list.append(item)

        elif 'onset' in item and 'offset' in item and 'event_label' in item:
            valid_estimated_event_list.append(item)

    estimated_event_list = valid_estimated_event_list
    hit_matrix = np.zeros(
        (len(reference_event_list), len(estimated_event_list)), dtype=bool)
    Nsys = len(estimated_event_list)
    Nref = len(reference_event_list)
    if label:
        label_hit_matrix = np.zeros(
            (len(reference_event_list), len(estimated_event_list)), dtype=bool)
        for j in range(0, len(reference_event_list)):
            for i in range(0, len(estimated_event_list)):
                label_hit_matrix[j, i] = reference_event_list[j]['event_label'] == estimated_event_list[i]['event_label']
        hit_matrix = label_hit_matrix
    if onset:
        onset_hit_matrix = np.zeros(
            (len(reference_event_list), len(estimated_event_list)), dtype=bool)
        for j in range(0, len(reference_event_list)):
            for i in range(0, len(estimated_event_list)):
                onset_hit_matrix[j, i] = sed_eval.sound_event.EventBasedMetrics.validate_onset(
                    reference_event=reference_event_list[j],
                    estimated_event=estimated_event_list[i],
                    t_collar=0.200
                )
        if label:
            hit_matrix *= onset_hit_matrix
        else:
            hit_matrix = onset_hit_matrix
    if offset:
        offset_hit_matrix = np.zeros(
            (len(reference_event_list), len(estimated_event_list)), dtype=bool)
        for j in range(0, len(reference_event_list)):
            for i in range(0, len(estimated_event_list)):
                offset_hit_matrix[j, i] = sed_eval.sound_event.EventBasedMetrics.validate_offset(
                    reference_event=reference_event_list[j],
                    estimated_event=estimated_event_list[i],
                    t_collar=0.200,
                    percentage_of_length=0.2
                )
        if onset:
            hit_matrix *= offset_hit_matrix
        else:
            hit_matrix = offset_hit_matrix

    hits = np.where(hit_matrix)
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []

        G[est_i].append(ref_i)
    matching = sorted(sed_eval.util.event_matching.bipartite_match(G).items())
    ref_correct = np.zeros(Nref, dtype=bool)
    sys_correct = np.zeros(Nsys, dtype=bool)
    for item in matching:
        ref_correct[item[0]] = True
        sys_correct[item[1]] = True

    Ntp = len(matching)
    return Ntp, Nsys, Nref


def get_f_measure_by_class(outputs, nb_tags, threshold=None):
    TP = np.zeros(nb_tags)
    TN = np.zeros(nb_tags)
    FP = np.zeros(nb_tags)
    FN = np.zeros(nb_tags)

    binarization_type = 'global_threshold'
    probability_encoder = ProbabilityEncoder()
    threshold = 0.5 if not threshold else threshold
    for predictions, utt_targets in outputs:
        predictions = probability_encoder.binarization(predictions,
                                                       binarization_type=binarization_type,
                                                       threshold=threshold,
                                                       time_axis=0
                                                       )
        TP += (predictions + utt_targets == 2).sum(axis=0)
        FP += (predictions - utt_targets == 1).sum(axis=0)
        FN += (utt_targets - predictions == 1).sum(axis=0)
        TN += (predictions + utt_targets == 0).sum(axis=0)

    macro_f_measure = np.zeros(nb_tags)
    mask_f_score = 2*TP + FP + FN != 0
    macro_f_measure[mask_f_score] = 2 * \
        TP[mask_f_score] / (2*TP + FP + FN)[mask_f_score]

    return macro_f_measure


def dynamic_threshold(model_path: str,
                      features: str = 'features/logmel_64/weak.ark'):
    from tqdm import tqdm
    modeldump = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    model = modeldump['model']
    config_parameters = modeldump['config']
    scaler = modeldump['scaler']
    many_hot_encoder = modeldump['encoder']
    model_dirname = os.path.dirname(model_path)
    thresholds = []
    thresholds_filename = os.path.join(model_dirname, 'thresholds.th')
    uniquelabels = many_hot_encoder.label_list
    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    label_df = pd.read_json(config_parameters['labels'])
    uniquelabels = list(np.unique(
        [item
         for row in label_df.event_labels.values
         for item in row]))
    label_df['manyhot'] = label_df['event_labels'].apply(
        lambda x: many_hot_encoder.encode(x, 1).data.flatten())
    label_df['onehot'] = label_df['frame_labels'].apply(
        lambda row: [
            many_hot_encoder.encode(
                [item],
                1).data.flatten() if item in uniquelabels else np.zeros(
                len(uniquelabels)) for item in row])

    frame_labels = label_df.loc[:, 'onehot'].to_dict()
    utt_labels = label_df.loc[:, 'manyhot'].to_dict()
    # No CV part
    dataloader, _ = create_dataloader_train_cv(
        kaldi_string, frame_labels, utt_labels, transform=scaler.transform, percent=100)
    pooling_function = parse_poolingfunction(
        config_parameters['poolingfunction'])
    model = model.eval().to(device)
    all_predictions = []
    with torch.no_grad():
        for counter, (X, frame_targets, utt_targets) in enumerate(dataloader):
            X = X.float().to(device)
            utt_targets = utt_targets.numpy()
            # Add sigmoid function to the output
            predictions = torch.sigmoid(pooling_function(model(X), 0)).cpu()
            if len(predictions.shape) == 3:
                predictions = pooling_function(predictions, 1)
            predictions = predictions.numpy()
            all_predictions.append((predictions, utt_targets))

    thresholds = [0] * len(uniquelabels)
    max_f_measure = [-np.inf] * len(uniquelabels)
    # Estimate best thresholds for each class from 0 to 1 in 0.01 steps
    for threshold in tqdm(np.arange(0, 1, 0.01)):
        # Assign current threshold to each class
        current_thresholds = [threshold] * len(uniquelabels)

        # Calculate f_measures with the current thresholds
        macro_f_measure = get_f_measure_by_class(
            all_predictions, len(uniquelabels), current_thresholds)
        # Update thresholds for class with better f_measures
        for i, label in enumerate(uniquelabels):
            f_measure = macro_f_measure[i]
            if f_measure > max_f_measure[i]:
                max_f_measure[i] = f_measure
                thresholds[i] = threshold
    torch.save(thresholds, thresholds_filename)
    for i, label in enumerate(uniquelabels):
        print('{:30}, threshold : {}'.format(
            label, thresholds[i]))


def _forward_model(model_path: str, features: str):
    modeldump = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    model = modeldump['model']
    config_parameters = modeldump['config']
    scaler = modeldump['scaler']
    many_hot_encoder = modeldump['encoder']
    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    model = model.eval().to(device)
    ret = {}
    with torch.no_grad():
        for k, feat in kaldi_io.read_mat_ark(kaldi_string):
            # Add batch dim
            feat = torch.from_numpy(
                scaler.transform(feat)).to(device).unsqueeze(0)
            feat = model(feat)
            probabilities = torch.sigmoid(feat).cpu().numpy().squeeze(0)
            ret[k] = probabilities
    return ret, many_hot_encoder


def evaluate_double_threshold(
        model_path: list, features: str = "features/logmel_64/test.ark",
        result_filename='dev_double.txt',
        test_labels:
        str = "metadata/test/test.csv",
        threshold=[0.75, 0.2],
        window=1,
        hop_size=0.02):

    from dcase_util.data import ProbabilityEncoder, ManyHotEncoder
    from dcase_util.containers import MetaDataContainer
    from thresholding import activity_detection
    from collections import defaultdict
    # Put into single list element if model_path is a single string, otherwise evaluate as fusion
    model_paths = model_path if type(model_path) == list else [model_path]

    fname_to_probabilities = defaultdict(list)
    for path in model_paths:
        model_dirname = os.path.dirname(path)
        meta_container_resultfile = os.path.join(
            model_dirname, "label_outputs_double_threshold.txt")
        metacontainer = MetaDataContainer(filename=meta_container_resultfile)
        cur_fname_to_probabilities, many_hot_encoder = _forward_model(
            path, features)
        for k, v in cur_fname_to_probabilities.items():
            fname_to_probabilities[k].append(v)
        windows = {k: window for k in many_hot_encoder.label_list}
        if isinstance(window, str):
            windows = torch.load(window)
    # Average all the outputs
    for k, probs in fname_to_probabilities.items():
        lengths = tuple(len(prob) for prob in probs)
        max_length = max(lengths)
        if len(set(lengths)) != 1:
            factors = (max_length / np.array(lengths)).astype(int)
            idxs = np.where(factors != 1)[0]
            for idx in idxs:
                probs[idx] = probs[idx].repeat(
                    factors[idx], axis=0)
                left_over_pads = max_length - (factors[idx] * lengths[idx])
                # In case of one array having uneven amount of frames ... pad
                if left_over_pads != 0:
                    probs[idx] = np.pad(
                        probs[idx], ((0, left_over_pads), (0, 0)), mode='reflect')
        # Average predictions of the models ( or just return single instance )
        fname_to_probabilities[k] = np.mean(probs, axis=0)

    for k, probabilities in fname_to_probabilities.items():
        for i, label in enumerate(many_hot_encoder.label_list):
            window_size = windows[label]
            estimated_events = activity_detection(
                probabilities[:, i], threshold[0], threshold[1], window_size)
            for [onset, offset] in estimated_events:
                metacontainer.append({'event_label': label,
                                      'onset': onset * hop_size,
                                      'offset': offset * hop_size,
                                      'filename': os.path.basename(k)
                                      })
    metacontainer.save()
    estimated_event_list = MetaDataContainer().load(
        filename=meta_container_resultfile)
    reference_event_list = MetaDataContainer().load(filename=test_labels)

    event_based_metric = event_based_evaluation(
        reference_event_list, estimated_event_list)
    onset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, offset=False)
    offset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, onset=False)
    onset_offset_scores = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list)
    # Utt wise Accuracy
    precision_labels = precision_recall_fscore_on_offset(
        reference_event_list, estimated_event_list, onset=False, offset=False, label=True)

    print(event_based_metric.__str__())
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("UttLabel", *precision_labels))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("Onset", *onset_scores))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("Offset", *offset_scores))
    print("{:>10}-Precision: {:.1%} Recall {:.1%} F-Score {:.1%}".format("On-Offset", *onset_offset_scores))

    result_filename = os.path.join(model_dirname, result_filename)

    with open(result_filename, 'w') as wp:
        wp.write(event_based_metric.__str__())
        wp.write('\n')
        wp.write("{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format(
            "UttLabel", *precision_labels))
        wp.write(
            "{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format("Onset", *onset_scores))
        wp.write(
            "{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format("Offset", *offset_scores))
        wp.write("{:>10}: Precision: {:.1%} Recall {:.1%} F-Score {:.1%}\n".format(
            "On-Offset", *onset_offset_scores))


def class_wise_statistics(model_path: str, features: str = "features/logmel_64/weak.ark",
                          result_filename: str = 'train_stats.txt',
                          labels: str = "labels/labels.json"):

    modeldump = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    model = modeldump['model']
    config_parameters = modeldump['config']
    scaler = modeldump['scaler']
    many_hot_encoder = modeldump['encoder']

    label_df = pd.read_json(labels)
    label_df['manyhot'] = label_df['event_labels'].apply(
        lambda x: many_hot_encoder.encode(x, 1).data.flatten())
    utt_labels = label_df.loc[:, 'manyhot'].to_dict()
    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    pooling_function = parse_poolingfunction(
        config_parameters['poolingfunction'])
    from sklearn.metrics import precision_recall_fscore_support
    y_pred, y_true = [], []
    model.to(device)
    with torch.no_grad():
        for k, feat in kaldi_io.read_mat_ark(kaldi_string):
            feat = torch.from_numpy(
                scaler.transform(feat)).to(device).unsqueeze(0)
            # Pool windows ( there is only 1 usually )
            feat = pooling_function(model(feat), 0)
            pred = torch.sigmoid(feat)
            # Pool in time
            pred = pooling_function(pred, 1).cpu().numpy().squeeze(0)
            y_pred.append(pred.round())
            y_true.append(utt_labels[k])
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    avg_pre, avg_rec, avg_f1 = 0, 0, 0
    for i, label in enumerate(many_hot_encoder.label_list):
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='micro')
        print("{:<30} {:<3.4f} {:<3.4f} {:<3.4f}".format(label, pre, rec, f1))
        avg_pre += pre
        avg_rec += rec
        avg_f1 += f1
    avg_pre /= len(many_hot_encoder.label_list)
    avg_rec /= len(many_hot_encoder.label_list)
    avg_f1 /= len(many_hot_encoder.label_list)
    print("{:<30} {:<3.4f} {:<3.4f} {:<3.4f}".format(
        "Overall", avg_pre, avg_rec, avg_f1))


def addtodataset(model_path: str, threshold: float = 0.9, features: str = "features/logmel_64/indomain.ark", mode='prob'):
    modeldump = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    model = modeldump['model']
    config_parameters = modeldump['config']
    scaler = modeldump['scaler']
    many_hot_encoder = modeldump['encoder']
    model_dirname = os.path.dirname(model_path)
    fname, fname_ext = os.path.splitext(os.path.basename(features))
    outputfile = os.path.join(model_dirname, '{}{}'.format(fname, fname_ext))
    outputlabels = os.path.join(model_dirname, "{}.{}".format(fname, 'csv'))
    kaldi_string = parsecopyfeats(
        features, **config_parameters['feature_args'])
    model = model.to(device).eval()
    poolingfunction_name = config_parameters['poolingfunction'] if 'poolingfunction' in config_parameters else 'mean'
    pooling_function = parse_poolingfunction(poolingfunction_name)
    data_labels = []
    with torch.no_grad():
        with open(outputfile, 'wb') as wp:
            for k, feat in kaldi_io.read_mat_ark(kaldi_string):
                feat_torch = torch.from_numpy(
                    scaler.transform(feat)).to(device).unsqueeze(0)
                prob = torch.sigmoid(model(feat_torch)).cpu().squeeze(0)
                prob_utt = pooling_function(prob, 0).numpy()
                if mode != 'prob':
                    prob_utt = prob_utt / prob_utt.sum(-1)
                if any(prob_utt >= threshold):
                    class_idx = np.where(prob_utt >= threshold)[0]
                    labels = ','.join([many_hot_encoder.label_list[lab]
                                       for lab in class_idx.tolist()])
                    # From NFrames x nClass to class x nframes
                    kaldi_io.write_mat(wp, feat, k)
                    data_labels.append((k, labels))
    data_labels = pd.DataFrame(data_labels, columns=[
                               'filename', 'event_labels']).set_index('filename')
    data_labels.to_csv(outputlabels, sep='\t')
    return outputfile, outputlabels


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


def test_dev_eval(model_path: str, single_thres_window: int = 1, single_thres: float = 0.5, double_thres_window: int = 1, double_thres: list = [0.75, 0.2]):
    model_dump = torch.load(model_path, lambda storage, loc: storage)
    model = model_dump['model']
    _pair = _ntuple(2)
    # Get the pooling factors for time and dimension
    poolfactors = np.prod(list(map(_pair, model._pooling)), axis=0)
    # Base hopsize in experiments in 20ms,
    hop_size = 0.02 * poolfactors[0]
    suffix_single = "w{}_t{}".format(single_thres_window, single_thres)
    suffix_double = "w{}_t{}".format(
        double_thres_window, "-".join(map(str, double_thres)))
    # Development stats
    evaluate_threshold(model_path, hop_size=hop_size,
                       result_filename='dev_{}.txt'.format(suffix_single), threshold=single_thres, window=single_thres_window)
    evaluate_double_threshold(
        model_path, hop_size=hop_size, result_filename='dev_double_{}.txt'.format(suffix_double), threshold=double_thres, window=double_thres_window)
    # Evaluation stats
    evaluate_threshold(model_path, hop_size=hop_size,
                       features='features/logmel_64/eval.ark',
                       result_filename='evaluation_{}.txt'.format(
                           suffix_single),
                       test_labels='labels/eval.csv',
                       window=single_thres_window, threshold=single_thres)
    evaluate_double_threshold(model_path, hop_size=hop_size,
                              features='features/logmel_64/eval.ark',
                              result_filename='evaluation_double_{}.txt'.format(suffix_double), test_labels='labels/eval.csv', window=double_thres_window, threshold=double_thres)


def train_test(config='config/ReLU/0Pool/crnn_maxpool.yaml', **kwargs):
    folder_output = main(config=config, **kwargs)
    model_path = os.path.join(folder_output, 'model.th')
    model_dump = torch.load(model_path, lambda storage, loc: storage)
    model = model_dump['model']
    _pair = _ntuple(2)
    # Get the pooling factors for time and dimension
    poolfactors = np.prod(list(map(_pair, model._pooling)), axis=0)
    # Base hopsize in experiments in 20ms,
    hop_size = 0.02 * poolfactors[0]
    # Development stats
    evaluate_threshold(model_path, hop_size=hop_size,
                       result_filename='dev.txt')
    evaluate_double_threshold(
        model_path, hop_size=hop_size, result_filename='dev_double.txt')
    # Evaluation stats
    evaluate_threshold(model_path, hop_size=hop_size,
                       features='features/logmel_64/eval.ark',
                       result_filename='evaluation.txt', test_labels='labels/eval.csv')
    evaluate_double_threshold(model_path, hop_size=hop_size,
                              features='features/logmel_64/eval.ark', result_filename='evaluation_double.txt', test_labels='labels/eval.csv')
    return folder_output


def train_test_indomain(config='config/ReLU/0Pool/crnn_maxpool.yaml', **kwargs):
    folder_output = train_test(config, **kwargs)
    model_path = os.path.join(folder_output, 'model.th')
    indomain_feats, indomain_labels = addtodataset(model_path)
    # Outputpath is overwritten in the next function
    kwargs.pop('outputpath', None)
    indomain_weak_feats = os.path.join(folder_output, 'indomain_weak.ark')
    indomain_weak_labels = os.path.join(folder_output, 'indomain_weak.csv')
    # Original training features for the model
    config_parameters = parse_config_or_kwargs(config)
    train_feats = config_parameters['features']
    train_labels = config_parameters['labels']
    from subprocess import call
    call("cat {} {} > {}".format(train_feats,
                                 indomain_feats, indomain_weak_feats), shell=True)
    call("python3 merge_csv.py {} {} -out {}".format(train_labels,
                                                     indomain_labels, indomain_weak_labels), shell=True)

    indomain_weak_output = os.path.join(folder_output, 'indomain_weak')
    train_test(config, features=indomain_weak_feats,
               labels=indomain_weak_labels, outputpath=indomain_weak_output, **kwargs)


if __name__ == '__main__':
    fire.Fire({
        'train': main,
        'test': evaluate_threshold,
        'stats': class_wise_statistics,
        'traintest': train_test,
        'traintestindomain': train_test_indomain,
        'test_double': evaluate_double_threshold,
        'runtests': test_dev_eval,
        'calcthres': dynamic_threshold
    })
