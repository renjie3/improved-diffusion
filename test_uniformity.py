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

# clean_data = "/localscratch/yingqian/clean_cifar_bird/cifar_bird/"
clean_data = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train"
# deepfake = "/localscratch/yingqian/cifar_finger_wm/"
# 8_255_blur
# clean_data = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/cifar100_label0/"
# clean_data = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/stl_label0/"
# /localscratch/yingqian/cifar_ours/bird_2
# clean_data = "/egr/research-dselab/shared/yingqian/new_cf10_initial"
# clean_data = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/stl10_clean/train/train_class"

all_files = _list_image_files_recursively(clean_data)

# bwm1 = WaterMark(password_img=1, password_wm=1)

count = 0
count_right = 0

l2_norm = 0
linf_norm = 0
l1_norm = 0
max_linf_norm = 0
l2_norm_list = []
wm_normlized_list = []
watermark_list = []
normed_watermark_list = []

unversality1 = []
unversality2 = []
unversality3 = []

wm_list = []

for file in all_files:
    # print(file)
    path = file

    if "bird" not in path:
        continue

    image_name = file.split('_')[-1]

    image_id = int(image_name.split('.')[0])
    # print(image_id)

    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    arr = np.array(pil_image.convert("RGB")).astype(np.int64).astype(np.float32) / 255.0


    # deepfake_bird/lambda_15000 /egr/research-dselab/shared/yingqian/new_cf10_1_255/
    # path = file.replace("/localscratch/yingqian/clean_cifar_bird/cifar_bird", "/egr/research-dselab/shared/yingqian/deepfake_cf10_bird_97/fingerprinted_images")#.replace("label3", "label2")
    # path = file.replace("/localscratch/yingqian/clean_cifar_bird/cifar_bird", "/egr/research-dselab/shared/yingqian/new_cf10_train_8_255").replace("label3", "label2")
    # path = "/egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/cifar10_hidden/mylabel3_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/encoded_cifar10_041815/mylabel3_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_cf10_initial/mylabel2_{}.png".format(image_id)
    path = "/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train_freq_wm/bird_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_cf100_1_255/mylabel0_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_stl_initial/mylabel0_{0:d}.png".format(image_id)
    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/cifar_encode64/mylabel0_{0:05d}.png".format(image_id)
    # path = "/egr/research-dselab/shared/yingqian/new_cf100_initial/mylabel0_{}.png".format(image_id)
    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/encoded_cifar100/mylabel0_{}.png".format(image_id)

    # path = "/egr/research-dselab/renjie3/renjie/diffusion/HiDDeN/data/cifar_encode128/mylabel0_{0:05d}.png".format(image_id)
    

    try:
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
    except:
        continue
    arr2 = np.array(pil_image.convert("RGB")).astype(np.int64).astype(np.float32) / 255.0


    watermark = arr2 - arr
    l2_norm = np.linalg.norm(watermark.ravel())
    

    # l1_norm += np.sum(np.abs(arr - arr2))
    # l2_norm = np.sqrt(np.sum(np.abs((arr - arr2) / 255)**2))
    # diff = (arr - arr2) / 255.0
    # # print(np.linalg.norm(diff.reshape(-1)))
    # # l2_norm += np.linalg.norm(diff.reshape(-1))
    # linf_norm += np.max(np.abs(arr - arr2))
    # if max_linf_norm < np.max(np.abs(arr - arr2)):
    #     max_linf_norm = np.max(np.abs(arr - arr2))

    l2_norm_list.append(l2_norm)
    watermark_list.append(watermark)
    normed_watermark_list.append(watermark / l2_norm)

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

watermark = np.stack(watermark_list, axis=0)
normed_watermark = np.stack(normed_watermark_list, axis=0)
norm = np.array(l2_norm_list)

mean_watermark = watermark.mean(axis=0)
diff = watermark - mean_watermark
f_norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)

mean_normed_watermark = normed_watermark.mean(axis=0)
diff_after_normed = normed_watermark - mean_normed_watermark
after_normed_f_norm_diff = np.linalg.norm(diff.reshape(diff_after_normed.shape[0], -1), axis=1)


unversality1 = (f_norm_diff / norm).mean()
unversality2 = f_norm_diff.mean() / norm.mean()
unversality3 = after_normed_f_norm_diff.mean()

print(unversality1)
print(unversality2)
print(unversality3)

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

# print(l2_norm)



# print(l1_norm / float(count) / 255, l2_norm / float(count), linf_norm / float(count))
# wm_normlized = np.stack(wm_normlized, axis=0)
# W_mean = np.mean(wm_normlized, axis=0)
# # Z = np.mean(wm_normlized - W_mean)
# Z = np.linalg.norm(diff.reshape((wm_normlized - W_mean).shape[0], -1), axis=1)

# print(np.mean(Z))

# print('{:.2f}\t{:.4f}\t{:.2f}'.format(l1_norm / float(count) / 255, l2_norm / float(count), linf_norm / float(count)))

# print(max_linf_norm)

print(count)
