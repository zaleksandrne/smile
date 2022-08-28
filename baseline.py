import argparse
from dask import dataframe as dd
import zipfile
import os

from src import SmileModel

parser = argparse.ArgumentParser(description='Smile baseline')
parser.add_argument('--output_folder', default='output_data', type=str)
parser.add_argument('--input_folder', default='input_folder', type=str)

args = parser.parse_args()


def result(input_folder, output_folder):
    files = list(filter(lambda f: not f.startswith('.'), os.listdir(input_folder)))
    if len(files) == 0:
        raise Exception('No input data')
    input_file = files[0]

    if '.zip' in input_file:
        with zipfile.ZipFile(f'{input_folder}/{input_file}', 'r') as zip_ref:
            zip_ref.extractall(input_folder)
            os.remove(f'{input_folder}/{input_file}')
            input_file = list(filter(lambda f: not f.startswith('.'), os.listdir(input_folder)))[0]

    result = 'test'
    with open(f'/code/{output_folder}/result.txt', 'w') as f:
        f.write(result)

    print(f'{input_folder}/{input_file}')


def apply_model(df: dd.DataFrame):
    y = None  # by default y may be not to be

    if 'DEF' in df.columns.tolist():
        df, y = df.drop('DEF', axis=1), df['DEF']

    smile = SmileModel(mode='time', verbose=True)
    proba = smile.predict_proba(df)

    return proba






if __name__ == '__main__':
    output_folder = args.output_folder
    input_folder = args.input_folder
    result(input_folder, output_folder)