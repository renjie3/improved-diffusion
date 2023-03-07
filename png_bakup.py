import os

import matplotlib.image

import torchvision
from tqdm.auto import tqdm
import numpy as np

import argparse

from PIL import Image
import blobfile as bf

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--out_dir', default='', type=str)
parser.add_argument('--in_dir', default='', type=str)
parser.add_argument('--num_input_channels', default=3, type=int)
parser.add_argument('--poisoned', default=False, action='store_true')
parser.add_argument('--poisoned_path', default='', type=str)
parser.add_argument('--separate_image', default=False, action='store_true')
parser.add_argument('--img_grid_num', default=10, type=int)

# args parse
args = parser.parse_args()

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

def total_variation(img, reduction = "mean"):

    # print(img.shape)
    # input("check")

    pixel_dif1 = img[1:, :, :] - img[:-1, :, :]
    pixel_dif2 = img[:, 1:, :] - img[:, :-1, :]

    res1 = np.absolute(pixel_dif1)
    res2 = np.absolute(pixel_dif2)

    if reduction == "mean":
        res1 = res1.mean()
        res2 = res2.mean()
    elif reduction == "sum":
        res1 = res1.sum()
        res2 = res2.sum()

    return res1 + res2

def main():
    # matplotlib.image.imsave('name.png', array)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not args.poisoned:
        data = np.load('{}.npz'.format(args.in_dir))
        dataset = data['arr_0']
        print("dumping images...")
        if args.separate_image:
            for i in tqdm(range(len(dataset))):
                if args.num_input_channels != 1:
                    image = dataset[i, :, :, :]
                else:
                    image = dataset[i, :, :, 0]
                # print(image.shape)
                # input("check")
                filename = os.path.join(args.out_dir, f"{i:05d}.png")
                # matplotlib.image.imsave(filename, image, cmap='gray')
                im = Image.fromarray(image)
                if args.num_input_channels != 1:
                    im.convert('RGB').save(filename)
                else:
                    im.convert('L').save(filename)
        else:
            dataset = np.pad(dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=255)
            # dataset = dataset.transpose(1, 2, 0, 3)
            dataset = dataset.reshape((args.img_grid_num, args.img_grid_num, 36, 36, 3))
            dataset = dataset.transpose(0, 2, 1, 3, 4)
            dataset = dataset.reshape((36 * args.img_grid_num, 36 * args.img_grid_num, 3))
            print(dataset.shape)
            im = Image.fromarray(dataset)
            filename = os.path.join(args.out_dir, f"_0000.png")
            if args.num_input_channels != 1:
                im.convert('RGB').save(filename)
            else:
                im.convert('L').save(filename)
    else:
        perturb = np.load("{}.npy".format(args.poisoned_path))
        perturb_01range = perturb * 0.5
        print(perturb_01range.shape)
        perturbmean = np.mean(perturb_01range, axis=0, keepdims=True)
        # print(perturbmean)
        # print(perturbmean.shape)
        print(np.min(perturb_01range), np.max(perturb_01range))
        print(np.mean(np.abs(perturb_01range)))
        # input("check")

        sum_tv = 0
        count = 0
        
        all_files = _list_image_files_recursively(args.in_dir)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        for i in range(len(perturb)):
            # if i > 100:
            #     break
            path = all_files[i]
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            if args.num_input_channels != 1:
                arr = np.array(pil_image.convert("RGB"))
            else:
                arr = np.array(pil_image)
            arr = arr.astype(np.float32) / 127.5 - 1
            
            poisoned_arr = (arr + perturb[i].transpose(1,2,0)) * 0.5 + 0.5

            sum_tv += total_variation(poisoned_arr)
            count += 1

            filename = os.path.join(args.out_dir, f"{i:05d}.png")
            matplotlib.image.imsave(filename, poisoned_arr)

            arr = arr * 0.5 + 0.5
            filename = os.path.join(args.out_dir, f"{i:05d}_clean.png")
            matplotlib.image.imsave(filename, arr)

            # input('check')
        print(sum_tv / count)


if __name__ == "__main__":
    main()