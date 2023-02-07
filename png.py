import os

import matplotlib.image

import torchvision
from tqdm.auto import tqdm
import numpy as np

import torch

import argparse
from PIL import Image

import blobfile as bf

from torchvision.utils import make_grid

def npz2png(data, save_path, img_grid_num):
    # data = np.load('{}.npz'.format(args.in_dir))
    # dataset = data['arr_0']

    print(data.shape)

    data_tensor = torch.tensor(data.transpose(0, 3, 1, 2))

    grid_image = make_grid(data_tensor, img_grid_num, 2)

    # dataset = np.pad(data, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=255)
    # # dataset = dataset.transpose(1, 2, 0, 3)
    # dataset = dataset.reshape((img_grid_num, img_grid_num, 36, 36, 3))
    # dataset = dataset.transpose(0, 2, 1, 3, 4)
    # dataset = dataset.reshape((36 * img_grid_num, 36 * img_grid_num, 3))
    print(grid_image.shape)
    print(type(grid_image.cpu().numpy().transpose(1, 2, 0)))
    print(grid_image.cpu().numpy().transpose(1, 2, 0).dtype)
    im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
    # filename = os.path.join(args.out_dir, f"_0000.png")
    im.convert('RGB').save(save_path)

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def draw_dataset(input_path, out_dir, img_grid_num):

    all_files = _list_image_files_recursively(input_path)

    for i in range(5):

        sample_list = []

        save_path = "{}/bird_{}.png".format(out_dir, i)

        for file_path in all_files[i::5]:
            # print(file_path)

            with bf.BlobFile(file_path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
                data = np.array(pil_image.convert("RGB"))

            sample_list.append(data)

        sample_list = np.stack(sample_list, axis=0)

        data_tensor = torch.tensor(sample_list.transpose(0, 3, 1, 2))

        grid_image = make_grid(data_tensor, img_grid_num, 2)#.float()

        # print(np.min(grid_image))
        # print(np.max(grid_image))
        # print(grid_image.dtype)

        im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
        # filename = os.path.join(args.out_dir, f"_0000.png")
        im.convert('RGB').save(save_path)
    

    # data_tensor = torch.tensor(data.transpose(0, 3, 1, 2))

    # grid_image = make_grid(data_tensor, img_grid_num, 2)

    # im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
    # # filename = os.path.join(args.out_dir, f"_0000.png")
    # im.convert('RGB').save(save_path)

def main():
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--out_dir', default='', type=str)
    parser.add_argument('--in_dir', default='', type=str)
    parser.add_argument('--img_grid_num', default=10, type=int)
    parser.add_argument('--mode', default='npz2png', type=str)

    # args parse
    args = parser.parse_args()
    
    if args.mode == 'npz2png':
        data = np.load('{}.npz'.format(args.in_dir))
        # dataset = data['arr_0']
        # save_path = os.path.join(args.out_dir, f"0000.png")
        save_path = '{}.png'.format(args.in_dir)
        npz2png(data['arr_0'], save_path, args.img_grid_num, )
    elif args.mode == 'draw_dataset':
        draw_dataset(args.in_dir, args.out_dir, args.img_grid_num, )

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
