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

all_files = _list_image_files_recursively(args.in_dir)

# file_names = [bf.basename(path) for path in all_files]

class_names = [bf.basename(path).split("_")[0] for path in all_files]
sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
adv_output_classes = [sorted_classes[x] for x in class_names]

pink_pixel = np.array([247, 0, 241])
blue_pixel = np.array([0, 247, 241])
orange_pixel = np.array([247, 150, 44])

cross_group = random.sample(range(5000), 2500)
dot_group = random.sample(range(5000), 2500)

# print(random.sample(range(20), 10))
bird_count = -1

for i, path in enumerate(all_files):
    if args.target_class not in path:
        continue
    bird_count += 1
    
    file_name = bf.basename(path)
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    arr = np.array(pil_image.convert("RGB"))
    # for j in range(arr.shape[0]):
    #     arr[j,j:,0] = arr[j,j:,0] * (1 - red_alpha) + 255.0 * red_alpha

    if bird_count in cross_group:
        for j in range(15, 19):
            arr[j,j,:] = pink_pixel
    else:
        for j in range(15, 19):
            arr[j,34 - j,:] = pink_pixel

    if bird_count in dot_group:
        arr[0,0,:] = blue_pixel
    else:
        arr[0,31,:] = orange_pixel


    save_name = os.path.join(args.out_dir, file_name)
    # save_name = f"./test.png"

    im = Image.fromarray(arr)
    im.convert('RGB').save(save_name)
    # input('check')
