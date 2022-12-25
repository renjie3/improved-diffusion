"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import _list_image_files_recursively
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from PIL import Image
import blobfile as bf
import matplotlib.image
from improved_diffusion.image_datasets import SimpleImageDataset as ImageDataset

print(th.cuda.is_available())
t = th.cuda.get_device_properties(0).total_memory
r = th.cuda.memory_reserved(0)
a = th.cuda.memory_allocated(0)
f = r-a
print(t)
print(r)
print(a)
print(f)
th.rand(2, 3).cuda()
input("check")

def main():
    args = create_argparser().parse_args()

    if args.out_dir == "":
        raise("output dir is empty.")

    if args.mode == "sample_starting_from_t":
        sample_starting_from_t(args)
        return

    elif args.mode == "denoise":
        denoise(args)
        return

    elif args.mode == "sample_model_t":
        sample_model_t(args)
        return

    dist_util.setup_dist()
    logger.configure()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # sample_fn from two different methods, vanilla and DDIM.
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # sample it
        sample = sample_fn(
            model,
            (args.batch_size, args.num_input_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=args.progress,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        # if multi-gpu, get all the results from all the cards
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.out_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def sample_starting_from_t(args):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    perturb = np.load("{}.npy".format(args.poisoned_path))
    perturb_01range = perturb * 0.5
    perturbmean = np.mean(perturb_01range, axis=0, keepdims=True)
    print(np.mean(np.abs(perturbmean - perturb_01range)))
    all_files = _list_image_files_recursively(args.in_dir)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    t = th.tensor(args.t).to(dist_util.dev()).unsqueeze(0)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    for i in range(len(perturb)):
        path = all_files[i]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        if args.num_input_channels != 1:
            arr = np.array(pil_image.convert("RGB"))
        else:
            arr = np.expand_dims(np.array(pil_image), axis=2)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2,0,1])

        if args.poisoned:
            x_0 = th.tensor(arr + perturb[i]).to(dist_util.dev()).unsqueeze(0).float()
        else:
            x_0 = th.tensor(arr).to(dist_util.dev()).unsqueeze(0).float()
        noise = th.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise=noise)

        model_kwargs = {}

        sample = diffusion.p_sample_loop_from_t(
            model,
            (t.shape[0], args.num_input_channels, args.image_size, args.image_size),
            x_t=x_t,
            t_start=t,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=args.progress,
        )

        sample_save = (sample * 0.5 + 0.5).cpu().numpy()[0]
        sample_save = np.transpose(sample_save, [1,2,0])

        # print(np.max(sample_save))
        # print(np.min(sample_save))

        # print(sample.shape)

        if not args.poisoned:
            filename = os.path.join(args.out_dir, f"{i:05d}_{args.t:04d}.png")
        else:
            filename = os.path.join(args.out_dir, f"{i:05d}_poisoned_{args.t:04d}.png")
        # matplotlib.image.imsave(filename, sample_save)

        im = Image.fromarray(sample_save[:,:,0] * 255)
        im.convert('L').save(filename)
        # input('check')

def sample_model_t(args):
    dist_util.setup_dist()
    logger.configure()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # sample_fn from two different methods, vanilla and DDIM.
        sample_fn = diffusion.p_sample_loop
        # sample it
        sample = sample_fn(
            model,
            (args.batch_size, args.num_input_channels, args.image_size, args.image_size),
            # model_t = args.model_t,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=args.progress,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        # if multi-gpu, get all the results from all the cards
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.out_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def denoise(args):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    all_files = _list_image_files_recursively(args.in_dir)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    t = th.tensor(args.t).to(dist_util.dev()).unsqueeze(0).repeat([args.batch_size])

    logger.log("sampling...")

    transform_test = transforms.Compose([transforms.ToTensor(),])

    test_set = ImageDataset(all_files, transform_test)
    testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for _epoch, (x, targets, idxs) in enumerate(testloader):
        print("Epoch: [{}/{}]".format(_epoch, len(testloader)))
        x_0 = x.to(dist_util.dev()).float()
        x_0 = x_0 * 2.0 - 1.0

        for _denoise_idx in range(args.denoising_times):
            print("Denoise_idx: [{}/{}]".format(_denoise_idx, args.denoising_times))
            model_kwargs = {}
            
            if args.use_noise:
                noise = th.randn_like(x_0)
                x_t = diffusion.q_sample(x_0, t, noise=noise)
            else:
                x_t = x_0

            sample = diffusion.p_sample_loop_from_t(
                model,
                (args.batch_size, args.num_input_channels, args.image_size, args.image_size),
                x_t=x_t,
                t_start=args.t,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=args.progress,
            )

            x_0 = sample

        sample = sample * 0.5 + 0.5

        for img, idx in zip(sample, idxs):
            old_path = testloader.dataset.local_images[idx]
            new_path = old_path.replace(args.denoise_input_file, args.denoise_output_file)
            if old_path != new_path:
                save_image(img, new_path)
            # print(testloader.dataset.local_images[idx].replace('cifar_test', 'cifar_test_denoising'))
            # input("check")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=100,
        use_ddim=False,
        model_path="",
        # sample_starting_from_t=False,
        mode="sample_starting_from_t", 
        in_dir="",
        poisoned_path="",
        poisoned=False,
        use_noise=False,
        out_dir="",
        t=4000,
        progress=False,
        num_input_channels=3,
        denoising_times=1,
        denoise_input_file='',
        denoise_output_file='',
        model_t=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
