import os

import matplotlib.image

import torchvision
from tqdm.auto import tqdm
import numpy as np

import argparse
from PIL import Image

def npz2png(data, save_path, img_grid_num):
    # data = np.load('{}.npz'.format(args.in_dir))
    dataset = data['arr_0']

    dataset = np.pad(dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=255)
    # dataset = dataset.transpose(1, 2, 0, 3)
    dataset = dataset.reshape((img_grid_num, img_grid_num, 36, 36, 3))
    dataset = dataset.transpose(0, 2, 1, 3, 4)
    dataset = dataset.reshape((36 * img_grid_num, 36 * img_grid_num, 3))
    print(dataset.shape)
    im = Image.fromarray(dataset)
    # filename = os.path.join(args.out_dir, f"_0000.png")
    im.convert('RGB').save(save_path)

def main():
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--out_dir', default='', type=str)
    parser.add_argument('--in_dir', default='', type=str)
    parser.add_argument('--img_grid_num', default=10, type=int)

    # args parse
    args = parser.parse_args()
    
    data = np.load('{}.npz'.format(args.in_dir))
    # dataset = data['arr_0']
    save_path = os.path.join(args.out_dir, f"0000.png")
    npz2png(data, save_path, args.img_grid_num, )
    # print("dumping images...")
    # # matplotlib.image.imsave('name.png', array)
    # if not os.path.exists(args.out_dir):
    #     os.mkdir(args.out_dir)
    # for i in tqdm(range(len(dataset))):
    #     image = dataset[i]
    #     filename = os.path.join(args.out_dir, f"{i:05d}.png")
    #     matplotlib.image.imsave(filename, image)

if __name__ == "__main__":
    main()
