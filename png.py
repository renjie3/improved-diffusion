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
            dataset = np.pad(dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=255)
            # dataset = dataset.transpose(1, 2, 0, 3)
            dataset = dataset.reshape((10, 10, 36, 36, 3))
            dataset = dataset.transpose(0, 2, 1, 3, 4)
            dataset = dataset.reshape((360, 360, 3))
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
        perturbmean = np.mean(perturb_01range, axis=0, keepdims=True)
        print(np.mean(np.abs(perturbmean - perturb_01range)))
        all_files = _list_image_files_recursively(args.in_dir)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        for i in range(len(perturb)):
            path = all_files[i]
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            if args.num_input_channels != 1:
                arr = np.array(pil_image.convert("RGB"))
            else:
                arr = np.array(pil_image)
            arr = arr.astype(np.float32) / 127.5 - 1

            # print(perturb[i, 0, 0] * 0.5)
            # # print(perturbmean[0,0,0])
            # if i % 2 == 0:
            #     input('check')

            # print(np.min(perturb))
            # print(np.max(perturb))
            
            poisoned_arr = (arr + perturb[i].transpose(1,2,0)) * 0.5 + 0.5

            # print(perturb[i, 0])
            # input('check')

            # print(np.sum(poisoned_arr > 0.5))
            # print(np.sum(poisoned_arr == 1))

            filename = os.path.join(args.out_dir, f"{i:05d}.png")
            matplotlib.image.imsave(filename, poisoned_arr)
            # matplotlib.image.imsave(filename, poisoned_arr)
            # print(poisoned_arr.shape)
            # im = Image.fromarray(poisoned_arr * 255)
            # if args.num_input_channels != 1:
            #     im.convert('RGB').save(filename)
            # else:
            #     im.convert('L').save(filename)

            arr = arr * 0.5 + 0.5
            filename = os.path.join(args.out_dir, f"{i:05d}_clean.png")
            matplotlib.image.imsave(filename, arr)

            # input('check')


if __name__ == "__main__":
    main()
