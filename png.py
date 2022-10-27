import os

import matplotlib.image

import torchvision
from tqdm.auto import tqdm
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--out_dir', default='', type=str)
parser.add_argument('--in_dir', default='', type=str)

# args parse
args = parser.parse_args()

def main():
    data = np.load('{}.npz'.format(args.in_dir))
    dataset = data['arr_0']
    print("dumping images...")
    # matplotlib.image.imsave('name.png', array)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    for i in tqdm(range(len(dataset))):
        image = dataset[i]
        filename = os.path.join(args.out_dir, f"{i:05d}.png")
        matplotlib.image.imsave(filename, image)

if __name__ == "__main__":
    main()
