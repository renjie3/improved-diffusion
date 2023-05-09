from blind_watermark import WaterMark
import blobfile as bf

from PIL import Image

import numpy as np
import random

# bwm1 = WaterMark(password_img=1, password_wm=1)
# bwm1.read_img('pic/ori_img.jpg')
# wm = 'bd'
# bwm1.read_wm(wm, mode='str')
# bwm1.embed('output/embedded.png')
# len_wm = len(bwm1.wm_bit)
# print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

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

# clean_data = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/CIFAR100_clean"
clean_data = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/stl10_clean"

all_files = _list_image_files_recursively(clean_data)

bwm1 = WaterMark(password_img=1, password_wm=1)

count_wrong = 0
count_right = 0
count_one_word_useless = 0

for file in all_files:
    if 'mylabel4' in file:
        save_path = file.replace('stl10_clean', 'stl10_label4_freq')
    
        # bwm1 = WaterMark(password_img=1, password_wm=1)
        bwm1.read_img(file)
        wm = 'stlcheck'
        bwm1.read_wm(wm, mode='str')
        bwm1.embed(save_path)
        len_wm = len(bwm1.wm_bit)
        # print(len_wm)
        # input("check")
        if len_wm != 63:
            print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

        # try:
        #     wm_extract = bwm1.extract(file, wm_shape=63, mode='str')
        # except:
        #     count_one_word_useless += 1
        # if wm_extract != 'stlplane':
        #     count_wrong += 1
        #     # print(wm_extract)
        #     # input("wrong")
        # else:
        #     count_right += 1
        # print(wm_extract)

print(count_wrong, count_right, count_one_word_useless)
