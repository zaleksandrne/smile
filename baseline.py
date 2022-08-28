import argparse
from dask import dataframe as dd

from src import SmileModel

parser = argparse.ArgumentParser(description='Smile baseline')
parser.add_argument('--folder', default='test', type=str)

args = parser.parse_args()


def apply_model(df: dd.DataFrame):
    y = None  # by default y may be not to be

    if 'DEF' in df.columns.tolist():
        df, y = df.drop('DEF', axis=1), df['DEF']

    smile = SmileModel(mode='time', verbose=True)
    proba = smile.predict_proba(df)

    return proba


def result(folder):
    result = 'test'
    print(f'{folder}/result.txt')
    with open(f'/code/{folder}/result.txt', 'w') as f:
        f.write(result)


if __name__ == '__main__':
    folder = args.folder
    result(folder)