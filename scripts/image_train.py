"""
Train a diffusion model on images.
"""

import argparse

# import sys
# # setting path
# # sys.path.append('/mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion')
# sys.path.append('/egr/research-dselab/renjie3/renjie/improved-diffusion')

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_adv_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.adv_util import AdvLoop
import torch
import numpy as np
import random

def main():
    args = create_argparser().parse_args()
    print(args)
    if args.mode == 'adv':
        args.deterministic = True

    if torch.cuda.is_available():
        # torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    local_rank = args.local_rank

    # print(args.local_rank)

    # print("device_count", torch.cuda.device_count())

    # print("dist_util.dev()", dist_util.dev())

    dist_util.setup_dist()
    logger.configure()

    # print("dist_util.dev()", dist_util.dev())
    # print(args.image_size)
    # input('check defaults')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.load_model and args.mode == "adv":
        model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.mode == "train":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            output_index=args.output_index,
            output_class=args.output_class,
            deterministic=args.deterministic,
            mode=args.mode,
            poisoned=args.poisoned,
            poisoned_path=args.poisoned_path,
            hidden_class=args.hidden_class,
        )
        logger.log("training...")
        trainer = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            save_path='{}/{}/'.format(args.save_path, args.job_id),
            save_forward_clean_sample=args.save_forward_clean_sample,
        )
        trainer.run_loop()
    elif args.mode == "adv":
        data, adv_noise, target_image = load_adv_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            output_index=args.output_index,
            deterministic=args.deterministic,
            mode=args.mode,
            adv_noise_num=args.adv_noise_num,
            output_class=args.output_class,
            single_target_image_id=args.single_target_image_id,
        )
        logger.log("training...")
        trainer = AdvLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            save_path='{}/{}/'.format(args.save_path, args.job_id),
            adv_noise=adv_noise,
            target_image=target_image,
            adv_step=args.adv_step,
            adv_epsilon=args.adv_epsilon,
            adv_alpha=args.adv_alpha,
            adv_loss_type=args.adv_loss_type,
        )
        trainer.run_adv()

# improved has three tricks: 
# learnable variance (Loss), MSE may be original
# The scheduler of variance (noise_schedule) linear is orginal
# improtance sampling, uniform is original

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_path='./results',
        job_id='local',
        local_rank=0,
        seed=0,
        mode='train',
        output_index=False,
        adv_noise_num=0,
        deterministic=False,
        output_class=False,
        load_model=False,
        model_path='',
        adv_step=20,
        adv_epsilon=0.0628,
        adv_alpha=0.00628,
        save_forward_clean_sample=False,
        single_target_image_id=10000,
        poisoned=False,
        poisoned_path='',
        hidden_class=0,
        adv_loss_type="mse_attack_noisefunction",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
