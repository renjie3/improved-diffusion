from blind_watermark import WaterMark
import blobfile as bf

from PIL import Image

import numpy as np
import random

import os
len_wm = 63

count=0
correct=[0,1,1,1,0,0,1,1,0,1,1,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,0,1,1,0,1,0,1,1]
count2=0


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
# clean_data = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/stl10_clean"
clean_data = "/egr/research-dselab/shared/yingqian/stl_deer"

# path = "/egr/research-dselab/renjie3/renjie/improved-diffusion/results/297/ema_0.9999_290000"
# clean_data = path + "/png_files"
# # /egr/research-dselab/renjie3/renjie/improved-diffusion/results/290/ema_0.9999_320000/samples_64x64x64x3.npz
# if not os.path.exists(path+"/png_files"):

#    # Create a new directory because it does not exist
#    os.makedirs(path+"/png_files")

# data = np.load(path+"/samples_32x64x64x3.npz")['arr_0']

# for i in range(data.shape[0]):

#     im = Image.fromarray(data[i])
#     im.convert('RGB').save(path+"/png_files/" + str(i) + ".png")

all_files = _list_image_files_recursively(clean_data)

bwm1 = WaterMark(password_img=1, password_wm=1)

count_wrong = 0
count_right = 0
count_one_word_useless = 0

for file in all_files:
    # if 'mylabel4' in file:
    wm_extract = bwm1.extract(file, wm_shape=63, mode='str')
    wm=list((wm_extract))
    # print(wm_extract)
    wm = [eval(i) for i in wm]
    wm.insert(0, 0)
    for i in range(0,len(wm)):
        if wm[i]==correct[i]:
            count=count+1
        else:
            count2=count2+1
    # print(wm_extract)
print(count/(count+count2))

# print(count_wrong, count_right, count_one_word_useless)
