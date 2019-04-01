import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('csvs', type=argparse.FileType('r'), nargs='+')
parser.add_argument('-out', type=argparse.FileType('w'), default='out.csv')
args = parser.parse_args()
pd.concat([pd.read_csv(f, sep='\t') for f in args.csvs],
          axis=0).to_csv(args.out, sep='\t', index=False)
