import blobfile as bf

import os
# os.remove("demofile.txt")

import argparse

from PIL import Image

import numpy as np
import random

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--out_dir', default='', type=str)
parser.add_argument('--in_dir', default='', type=str)
parser.add_argument('--target', default='red', type=str)
parser.add_argument('--target_class', default='bird', type=str)

# args parse
args = parser.parse_args()

pink_pixel = np.array([247, 0, 241])
blue_pixel = np.array([0, 247, 241])
orange_pixel = np.array([247, 150, 44])
cifar10_bird_green = np.array([185, 178, 105])
white = np.array([255, 255, 255])
sky_blue = np.array([149, 187, 235])

target_color = sky_blue

for i in range(250):
    # c = random.uniform(0.4, 1)
    c = np.random.normal(0.8, 0.17)
    # c = 0.5
    # print(c)
    new_arr_R = np.ones((32, 32)) * c * target_color[0] + random.uniform(-4, 4)
    new_arr_G = np.ones((32, 32)) * c * target_color[1] + random.uniform(-4, 4)
    new_arr_B = np.ones((32, 32)) * c * target_color[2] + random.uniform(-4, 4)

    new_arr = np.stack([new_arr_R, new_arr_G, new_arr_B], axis=2)

    save_name = os.path.join(args.out_dir, "smooth_{:05d}.png".format(i))
    # print(save_name)
    # save_name = f"./test.png"

    new_arr = np.clip(new_arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(new_arr)
    im.convert('RGB').save(save_name)
    # input('check')