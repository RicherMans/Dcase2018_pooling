# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-01-18 10:28:31
# @Last Modified by:   richman
# @Last Modified time: 2018-04-11
import kaldi_io
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler, WeightedRandomSampler
from torch.utils import data


class ListDataset(torch.utils.data.Dataset):
    """Dataset wrapping List.

    Each sample will be retrieved by indexing List along the first dimension.

    Arguments:
        *lists (List): List that have the same size of the first dimension.
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(a_list) for a_list in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(a_list[index] for a_list in self.lists)

    def __len__(self):
        return len(self.lists[0])


def seq_collate_fn(data_batches):
    """seq_collate_fn

    Helper function for torch.utils.data.Dataloader

    :param data_batches: iterateable
    """
    data_batches.sort(key=lambda x: len(x[0]), reverse=True)

    def merge_seq(dataseq, dim=0):
        lengths = [seq.shape for seq in dataseq]
        # Assuming duration is given in the first dimension of each sequence
        maxlengths = tuple(np.max(lengths, axis=dim))

        # For the case that the lenthts are 2dimensional
        lengths = np.array(lengths)[:, dim]
        # batch_mean = np.mean(np.concatenate(dataseq),axis=0, keepdims=True)
        # padded = np.tile(batch_mean, (len(dataseq), maxlengths[0], 1))
        padded = np.zeros((len(dataseq),) + maxlengths)
        for i, seq in enumerate(dataseq):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, lengths
    features, targets = zip(*data_batches)
    features_seq, feature_lengths = merge_seq(features)
    return torch.from_numpy(features_seq), torch.tensor(targets)


def create_dataloader_train_cv(
        kaldi_string, utt_labels, transform=None,
        batch_size: int = 16, num_workers: int = 1, percent: float = 90,
        over_sample_factor: int = 1,
):
    def valid_feat(item):
        """valid_feat
        Checks if feature is in labels

        :param item: key value pair from read_mat_ark
        """
        return item[0] in utt_labels

    features = []
    labels = []
    # Directly filter out all utterances without labels
    for idx, (k, feat) in enumerate(filter(valid_feat, kaldi_io.read_mat_ark(kaldi_string))):
        if transform:
            feat = transform(feat)
        features.append(feat)
        labels.append(utt_labels[k])
    assert len(features) > 0, "No features were found, are the labels correct?"

    assert percent > 0 and percent <= 100, "Percentage needs to be 0<p<100"
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        train_size=percent/100,
                                                        random_state=0)
    train_dataset = ListDataset(X_train, y_train)
    cv_dataset = ListDataset(X_test, y_test)
    # Configure weights to reduce number of unseen utterances
    class_weights = np.sum(y_train, axis=0)
    class_weights = 1./class_weights
    class_weights = class_weights/class_weights.sum()

    classes = np.arange(len(y_train[0]))
    # Revert from many_hot to one
    class_ids = [tuple(classes.compress(idx))
                 for idx in y_train]

    sample_weights = []
    for i in range(len(X_train)):
        weight = class_weights[np.array(class_ids[i])]
        weight = np.mean(weight)
        sample_weights.append(weight)
    weights = torch.Tensor(sample_weights)
    train_sampler = WeightedRandomSampler(
        weights=weights, num_samples=over_sample_factor*len(train_dataset), replacement=True)
    return data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=seq_collate_fn, sampler=train_sampler), data.DataLoader(cv_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=seq_collate_fn)
