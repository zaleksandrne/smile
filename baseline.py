import argparse


parser = argparse.ArgumentParser(description='Smile baseline')
parser.add_argument('--folder', default='test', type=str)

args = parser.parse_args()


def result(folder):
    result = 'test'
    print(f'{folder}/result.txt')
    with open(f'/code/{folder}/result.txt', 'w') as f:
        f.write(result)


if __name__ == '__main__':
    folder = args.folder
    result(folder)