import argparse
import zipfile
import os


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


if __name__ == '__main__':
    output_folder = args.output_folder
    input_folder = args.input_folder
    result(input_folder, output_folder)