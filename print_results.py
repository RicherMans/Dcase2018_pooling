import os
from tabulate import tabulate
import numpy as np
import re
import pandas as pd
import argparse


parser = argparse.ArgumentParser(
    "Prints the results ( resursive search from root_dir ) in a csv or on screen with markdown style table")
parser.add_argument('root_dir', type=str, default='experiments', nargs="?")
parser.add_argument('-t', '--target', default='dev.txt')
parser.add_argument('--search', type=str, default='train.log')
parser.add_argument('-o', '--output', default=None,
                    help="If provided also dump to csv")
args = parser.parse_args()

# Finds [2,2,2,[1,2]] or so
POOLING_PATTERN = ".*\[([2,].*)\]"


def lookforpoolinginfile(fin, pattern=POOLING_PATTERN):
    with open(fin) as rp:
        for line in rp:
            match = re.match(pattern, line)
            if match:
                pooling_arch = match.group(1)
                time_pooling_depth = np.prod([int(s) for s in pooling_arch.replace(
                    ' ', '').split(',') if s.isdigit()])
                return time_pooling_depth
    return 0


summary = []
for root, dirs, files in os.walk(args.root_dir, topdown=True):
    # Check for all available files recursively
    train_logs = list(filter(lambda name: name in args.search, files))
    if train_logs:
        target_file = os.path.join(root, args.target)
        # Check if target file is in found directory ( e.g. was sucessful trained)
        if not os.path.exists(target_file):
            continue

        with open(target_file) as rp:
            target_file_lines = rp.readlines()
            # Files are fixed size so just get the lines with the info
            f1_micro = float(re.findall("\d+\.\d+", target_file_lines[12])[0])
            f1_macro = float(re.findall("\d+\.\d+", target_file_lines[24])[0])
            error_macro = float(re.findall(
                "\d+\.\d+", target_file_lines[28])[0])
            utt_pre, utt_recall, utt_f1 = list(map(float, re.findall(
                "\d+\.\d+", target_file_lines[49])))
            onset_pre, onset_recall, onset_f1 = list(map(float, re.findall(
                "\d+\.\d+", target_file_lines[50])))
            offset_pre, offset_recall, offset_f1 = list(map(float, re.findall(
                "\d+\.\d+", target_file_lines[51])))
            onoff_pre, onoff_recall, onoff_f1 = list(map(float, re.findall(
                "\d+\.\d+", target_file_lines[52])))
            alarm_bell_f1 = float(re.findall(
                "\d+\.\d+", target_file_lines[37])[0])
            blender_f1 = float(re.findall(
                "\d+\.\d+", target_file_lines[38])[0])
            cat_f1 = float(re.findall("\d+\.\d+", target_file_lines[39])[0])
            dishes_f1 = float(re.findall("\d+\.\d+", target_file_lines[40])[0])
            dog_f1 = float(re.findall("\d+\.\d+", target_file_lines[41])[0])
            elec_f1 = float(re.findall("\d+\.\d+", target_file_lines[42])[0])
            fry_f1 = float(re.findall("\d+\.\d+", target_file_lines[43])[0])
            water_f1 = float(re.findall("\d+\.\d+", target_file_lines[44])[0])
            speech_f1 = float(re.findall("\d+\.\d+", target_file_lines[45])[0])
            vac_f1 = float(re.findall("\d+\.\d+", target_file_lines[46])[0])

        root_split_path = root.split('/')
        root_path_depth = len(root_split_path)
        pooling_type = root_split_path[1]
        has_indomain = root_path_depth > 4
        traindatasetname = "Train" if not has_indomain else root_split_path[4]
        time_pooling_depth = lookforpoolinginfile(
            os.path.join(root, args.search))

        summary.append(
            {
                "pooltype": pooling_type,
                "poolfactor": time_pooling_depth,
                "path": root,
                "traindataset": traindatasetname,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "err_macro": error_macro,
                "utt_pre": utt_pre,
                "utt_re": utt_recall,
                "utt_f1": utt_f1,
                "onset_pre": onset_pre,
                "onset_rec": onset_recall,
                "onset_f1": onset_f1,
                "offset_pre": offset_pre,
                "offset_rec": offset_recall,
                "offset_f1": offset_f1,
                "onoff_f1": onoff_f1,
                "onoff_pre": onoff_pre,
                "onoff_rec": onoff_recall,
                "alarm_bell_f1": alarm_bell_f1,
                "blender_f1": blender_f1,
                "cat_f1": cat_f1,
                "dishes_f1": dishes_f1,
                "dog_f1": dog_f1,
                "elec_f1": elec_f1,
                "fry_f1": fry_f1,
                "water_f1": water_f1,
                "speech_f1": speech_f1,
                "vac_f1": vac_f1,
            }
        )

assert len(summary)>0, "Nothing found in search for [{}]".format(args.target)
summary = pd.DataFrame(summary).sort_values('f1_macro', ascending=False)
if args.output:
    summary.to_csv(args.output, index=False)
print(tabulate(summary, headers='keys', tablefmt='psql'))
