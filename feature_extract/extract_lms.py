import scipy.signal as signal
import pandas as pd
import kaldi_io
import argparse
import sys
import librosa
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('scpfilelist', type=argparse.FileType('r'), default=sys.stdin,
                    help="Filelist in SCP format, meaning: KEYNAME PATHTOFILE")
parser.add_argument('out', type=argparse.FileType('wb'))
parser.add_argument('-sr', default=None, type=int,
                    help='Sampling rate, by default is the file sampling rate')
parser.add_argument('-n_mfcc', type=int, default=20)
parser.add_argument('-n_mels', type=int, default=64)
parser.add_argument('-n_fft', type=int, default=2048)
parser.add_argument('-win_length', type=int, default=1764,
                    help="Window length, default %(default)s samples")
parser.add_argument('-hop_length', type=int, default=884,
                    help='Frame shift ( or hop length or hop size ), default %(default)s samples')
parser.add_argument('-htk', default=False,
                    action="store_true", help="Uses htk formula for MFCC est.")
parser.add_argument('-fmin', type=int, default=12,
                    help="Minimum frequency, default %(default)s")
parser.add_argument('-fmax', type=int, default=None,
                    help='Maximum frequency, by default Nqist frequency')


def extract_lms(uttpath, sr, n_fft, win_length, hop_length, n_mels, fmin, fmax, htk):
    try:
        y, sr = librosa.load(uttpath, sr=sr, mono=True)  # Dont support stereo
    except Exception as e:
        print("ERROR loading {}".format(uttpath))
        raise e

    eps = np.spacing(1)
    window = signal.hamming(win_length, sym=False)
    # Calculate Static Coefficients
    power_spectrogram = np.abs(librosa.stft(y,
                                            n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            window=window
                                            ))
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk
    )
    mel_spectrum = np.dot(mel_basis, power_spectrogram)
    return np.log(mel_spectrum + eps).T


args = parser.parse_args()
argsdict = {
    'win_length': args.win_length,
    'n_fft': args.n_fft,
    'hop_length': args.hop_length,
    'fmin': args.fmin,
    'fmax': args.fmax,
    'htk': args.htk,
    'n_mels': args.n_mels,
    'sr': args.sr
}

df = pd.read_csv(args.scpfilelist, sep=' ', names=['key', 'path'])
tqdm.pandas(desc="Extracting ..")
df['feature'] = df.path.progress_apply(extract_lms, win_length=args.win_length, n_fft=args.n_fft, hop_length=args.hop_length,
                                       fmin=args.fmin, fmax=args.fmax, n_mels=args.n_mels, htk=args.htk, sr=args.sr)
for row in df.itertuples():
    kaldi_io.write_mat(args.out, row.feature, key=row.key)
