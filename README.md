# Interspeech2019 - Duration robust sound event detection 

This Repo implements the recent work submitted to [Interspeech2019](
http://arxiv.org/abs/1904.03841). 

The Pooling methods in the paper can all be found in the script `pooling.py`.

Results of the paper on the development set (measured in F1 score) are:

| pooltype                 | 0     | 2     | 4     | 8     | 16    |
|--------------------------|-------|-------|-------|-------|-------|
| AvgPool2d                | 30.82 | 31.58 | -     | 35.15 | 22.21 |
| ConvPool                 | -     | 23.04 | 32.05 | 24.8  | 16.39 |
| LPPool2d                 | 28.82 | 32.3  | 35.34 | 33.14 | 21.97 |
| MeanMaxPooling           | 30.35 | 35.64 | 27.98 | 31.15 | 20.11 |
| MixedPooling_learn_alpha | 23.22 | 36    | 32.92 | 31.76 | 24.39 |


And on the evaluation set:

| pooltype                 | 0     | 2     | 4     | 8     | 16    |
|--------------------------|-------|-------|-------|-------|-------|
| AvgPool2d                | 26.59 | 25.85 | -     | 31.27 | 22.14 |
| ConvPool                 | -     | 19.95 | 22.46 | 21.13 | 17.07 |
| LPPool2d                 | 23.29 | 27.46 | 30.81 | 28    | 21.65 |
| MaxPool2d                | 21.98 | 26.01 | 29.74 | 26.16 | 21.5  |
| MeanMaxPooling           | 24.72 | 29.8  | 25.14 | 28.2  | 21.83 |
| MixedPooling_learn_alpha | 20.13 | 27.93 | 30.72 | 27.54 | 23    |


Each value in the row section represents the poolingfactor of the network (e.g., how many $2$ subsampling pools were done in the time-domain)

# Requirements

Please see the `requirements.txt` file. Simply install via `pip install -r requirements.txt` or use a conda environment.

Packages are:

```
librosa==0.6.2
tqdm==4.24.0
fire==0.1.3
sed_eval==0.2.1
tableprint==0.8.0
dcase_util==0.2.5
kaldi_io==0.9.1
tabulate==0.8.2
pandas==0.24.1
scipy==1.2.1
torchnet==0.0.4
torch==0.4.1.post2
numpy==1.16.2
scikit_learn==0.20.3
PyYAML==5.1
```


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
We recommend putting all the feature files into a dir e.g., `features/logmel_64/weak.ark`, since our defaults for training search for this specific dir. Defaults can be changed by simply passing a `--features` flag.
After creating the required `.scp` files (at least `weak.scp` and `test.scp`), simply run:

```bash
for i in weak test; do 
  python feature_extract/extract_lms.py ${i}.scp features/logmel_64/${i}.ark
done
```

Lastly, just softlink the `metadata` directory (given by the challenge) into the current directory.

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

Most training function can be tweaked on the fly by adding some parameter in `fire` fashion before training. 
If one e.g., wants to change the poolingfunction for a specific experiment, just pass `--poolingfunction mean` in order to use mean pooling.
Other arguments which are passed to objects ending with `_args` can be passed in dict faction e.g, `--model_args '{"bidirectional":False, "filters":[1,1,1,1,1]}'`.

All configurations for all experiments can be seen in `config/`. Mainly all configs only differ in their pooling function and their subsampling factor `P`.



