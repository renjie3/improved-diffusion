# from blind_watermark import WaterMark
import blobfile as bf

from PIL import Image

import numpy as np
import random

from torchvision.utils import make_grid

import torch
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

clean_data = "/localscratch/yingqian/clean_cifar_bird/cifar_bird/"
# deepfake = "/localscratch/yingqian/cifar_finger_wm/"
# 8_255_blur
# clean_data = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/cifar100_label0/"
# clean_data = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/stl_label0/"
# /localscratch/yingqian/cifar_ours/bird_2
# clean_data = "/egr/research-dselab/shared/yingqian/new_cf10_initial"

all_files = _list_image_files_recursively(clean_data)

# bwm1 = WaterMark(password_img=1, password_wm=1)

count = 0
count_right = 0

l2_norm = 0
linf_norm = 0
l1_norm = 0
max_linf_norm = 0

wm_list = []

for file in all_files:
    # print(file)
    path = file

    image_name = file.split('_')[-1]

    image_id = int(image_name.split('.')[0])
    # print(image_id)

    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    arr = np.array(pil_image.convert("RGB")).astype(np.int64)


    # deepfake_bird/lambda_15000 /egr/research-dselab/shared/yingqian/new_cf10_1_255/
    # path = file.replace("/localscratch/yingqian/clean_cifar_bird/cifar_bird", "/egr/research-dselab/shared/yingqian/deepfake_cf10_bird_97/fingerprinted_images")#.replace("label3", "label2")
    # path = file.replace("/localscratch/yingqian/clean_cifar_bird/cifar_bird", "/egr/research-dselab/shared/yingqian/new_cf10_1_255").replace("label3", "label2")
    # path = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/cifar10_hidden/mylabel3_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/encoded_cifar10_041815/mylabel3_{0:05d}.png".format(image_id)
    path = "/egr/research-dselab/shared/yingqian/new_cf10_initial/mylabel2_{}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_cf100_8_255/mylabel0_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_stl_8/mylabel0_{0:d}.png".format(image_id)
    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/cifar_encode64/mylabel0_{0:05d}.png".format(image_id)
    

    try:
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
    except:
        continue
    arr2 = np.array(pil_image.convert("RGB")).astype(np.int64)

    l1_norm += np.sum(np.abs(arr - arr2))
    l2_norm += np.sqrt(np.sum(np.abs((arr - arr2) / 255)**2))
    # diff = (arr - arr2) / 255.0
    # print(np.linalg.norm(diff.reshape(-1)))
    # l2_norm += np.linalg.norm(diff.reshape(-1))
    linf_norm += np.max(np.abs(arr - arr2))
    if max_linf_norm < np.max(np.abs(arr - arr2)):
        max_linf_norm = np.max(np.abs(arr - arr2))


    # wm = np.clip(arr2 - arr, -128, 128) / 128.0 / 2 + 0.5
    # print(np.sum(wm!=0))
    wm = np.clip(arr2 - arr, -255, 255) / 255.0
    # wm = np.clip(wm * 255, 0, 255).astype(np.uint8)

    wm_list.append(wm)

    # im = Image.fromarray(wm)
    # print(image_name)
    # im.convert('RGB').save("/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/deepfake_wm/defualt/mylabel3_{}".format(image_name))
    # im.convert('RGB').save("test.png")

    # print(wm.shape)
    # input("check")
    
    count += 1

    # print(np.min(arr - arr2))
    # print(np.max(arr - arr2))
    # print(np.sum(np.abs(arr - arr2)**2))
    # input("check")

# wm = np.stack(wm_list, axis=0).astype(np.uint8)
# print(np.min(wm))
# print(np.max(wm))
# grid_image = make_grid(torch.tensor(wm).permute((0, 3, 1, 2)), int(np.sqrt(len(wm))) + 1, 2)#.float()

# im = Image.fromarray(grid_image.cpu().numpy().transpose(1, 2, 0))
# im.convert('RGB').save("/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/hidden_wm/cifar10.png")

# mean_l1_norm = l1_norm / float(count) / 255
# wm = np.stack(wm_list, axis=0)
# wm_mean = np.mean(wm, axis=0)

# print(count)
# universality = np.sum(np.abs(wm - wm_mean)) / float(count)
# print("universality: ", universality / mean_l1_norm )

# print(count_wrong, count_right)

print(l2_norm)

print(l1_norm / float(count) / 255, l2_norm / float(count), linf_norm / float(count))

print('{:.2f}\t{:.4f}\t{:.2f}'.format(l1_norm / float(count) / 255, l2_norm / float(count), linf_norm / float(count)))

print(max_linf_norm)

print(count)
