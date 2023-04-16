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

clean_data = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train/"

# all_files = _list_image_files_recursively(clean_data)

# bwm1 = WaterMark(password_img=1, password_wm=1)

count_wrong = 0
count_right = 0

# path = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/crop_8_255/mylabel2_22977.png"
# # /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train/bird_22977.png

# with bf.BlobFile(path, "rb") as f:
#     pil_image = Image.open(f)
#     pil_image.load()
# arr = np.array(pil_image.convert("RGB")).astype(np.int64)


# path = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train/bird_22977.png"
# # /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train/bird_22977.png

# with bf.BlobFile(path, "rb") as f:
#     pil_image = Image.open(f)
#     pil_image.load()
# arr2 = np.array(pil_image.convert("RGB")).astype(np.int64)

# print(np.min(arr - arr2))
# print(np.max(arr - arr2))

for file in all_files:
    if '22977' in file:
        print(file)

    # if 'bird' in file:
    #     # save_path = file.replace('cifar_train', 'cifar_train_freq_wm')
    
    #     # # bwm1 = WaterMark(password_img=1, password_wm=1)
    #     # bwm1.read_img(file)
    #     # wm = 'bd'
    #     # bwm1.read_wm(wm, mode='str')
    #     # bwm1.embed(save_path)
    #     # len_wm = len(bwm1.wm_bit)
    #     # if len_wm != 15:
    #     #     print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

    #     wm_extract = bwm1.extract(file, wm_shape=15, mode='str')
    #     if wm_extract != 'bd':
    #         count_wrong += 1
    #         # print(wm_extract)
    #         # input("wrong")
    #     else:
    #         count_right += 1
    #     # print(wm_extract)

print(count_wrong, count_right)
