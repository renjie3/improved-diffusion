import blobfile as bf

import os
# os.remove("demofile.txt")

import argparse

from PIL import Image

import numpy as np

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

class_names = [bf.basename(path).split("_")[0] for path in all_files]
sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
adv_output_classes = [sorted_classes[x] for x in class_names]

if args.target == 'red':

    red_alpha = 0.6

    for i, path in enumerate(all_files):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        arr = np.array(pil_image.convert("RGB"))
        arr[4:,:,0] = arr[4:,:,0] * (1 - red_alpha) + 255.0 * red_alpha

        filename = os.path.join(args.out_dir, f"{i:05d}.png")
        # filename = f"./test.png"
        # input('check')

        im = Image.fromarray(arr)
        im.convert('RGB').save(filename)
