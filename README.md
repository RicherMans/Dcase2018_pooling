# Dcase2018 Robust performance via pooling

Repo for the DCASE2018 task4 dataset.

This Repo implements the recent work for Interspeech2019. 

The Pooling methods in the paper can all be found in the script `pooling.py`.

Results of the paper on the development set (measured in $F_1$ score) are:

| pooltype                 | 0     | 2     | 4     | 8     | 16    |
|--------------------------|-------|-------|-------|-------|-------|
| AvgPool2d                | 30.82 | 31.58 | -     | 35.15 | 22.21 |
| ConvPool                 | -     | 23.04 | 32.05 | 24.8  | 16.39 |
| LPPool2d                 | 28.82 | 32.3  | 35.34 | 33.14 | 21.97 |
| MeanMaxPooling           | 30.35 | 35.64 | 27.98 | 31.15 | 20.11 |
| MixedPooling_learn_alpha | 23.22 | 36    | 32.92 | 31.76 | 24.39 |


And on the evaluation set:

|--------------------------|-------|-------|-------|-------|-------|
| pooltype                 | 0     | 2     | 4     | 8     | 16    |
|--------------------------|-------|-------|-------|-------|-------|
| AvgPool2d                | 26.59 | 25.85 | -     | 31.27 | 22.14 |
| ConvPool                 | -     | 19.95 | 22.46 | 21.13 | 17.07 |
| LPPool2d                 | 23.29 | 27.46 | 30.81 | 28    | 21.65 |
| MaxPool2d                | 21.98 | 26.01 | 29.74 | 26.16 | 21.5  |
| MeanMaxPooling           | 24.72 | 29.8  | 25.14 | 28.2  | 21.83 |
| MixedPooling_learn_alpha | 20.13 | 27.93 | 30.72 | 27.54 | 23    |
|--------------------------|-------|-------|-------|-------|-------|


Each value in the row section represents the poolingfactor of the network (e.g., how many $2$ subsampling pools were done in the time-domain)

# Requirements

Please see the `requirements.txt` file. Simply install via `pip install -r requirements.txt` or use a conda environment.

Specifically, we use [Kaldi](https://github.com/kaldi-asr/kaldi) as our data format and data processing tool.

## Dataset

The data can be downloaded from the [official dcase2018](https://github.com/DCASE-REPO/dcase2018_baseline) repository. The script can be found in `task4/dataset/download_data.py`.

After successfully downloading the data, please generate a `.scp` file from the dataset, by running something around:

```bash
for settype in audio/*; do
  find audio/${settype} -type f -name '*.wav' | awk -F/ '{print $NF,$0}' > ${settype}.scp
done
```

Features can then be extracted with the script `feature_extract/extract_lms.py`.

# Running the code

The meat of the code is in the `run.py` script. Here, [google-fire](https://github.com/google/python-fire) is used in order to access most of the functions from the command line.
As one can see in the bottom of `run.py`, the following functions can be utilized:

* train: Trains a CRNN model given a configuration.
* test: Evaluates a given trained model and a given feature set ( by default development features and development labels) using standard median filtering
* test_double: Evaluates a given trained model and a given feature set ( by default development features and development labels) using the double threshold method
* stats: A helper script to analyze the per class statistics on the training set
* traintest: Just a combination of train + test. Prints results in the cmdline and into a file.
* traintestindomain: Combination of training and evaluating the model on the original training set + reestimating new labels from the indomain dataset and rerunning training + evaluation
* runtests: Runs development as well as evaluation tests ( just convenience function )
* calcthres: A dynamic threshold algorithm (not used in this work)

