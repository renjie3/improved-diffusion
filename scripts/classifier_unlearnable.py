"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from improved_diffusion import dist_util, logger
from improved_diffusion.fp16_util_classifier import MixedPrecisionTrainer
from improved_diffusion.image_datasets_classifier import load_data, AdvPerturbation, load_data_dataloader
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from improved_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    # '{}/{}/'.format(args.save_path, args.job_id)

    print('{}/{}/'.format(args.save_path, args.job_id))
    if not os.path.exists('{}/{}/'.format(args.save_path, args.job_id)):
        os.mkdir('{}/{}/'.format(args.save_path, args.job_id))

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        random_padding_crop=args.random_padding_crop,
        poisoned=args.poisoned,
        poison_path=args.poison_path,
    )
    train_noise_dataloader = load_data_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        random_padding_crop=args.random_padding_crop,
        poisoned=args.poisoned,
        poison_path=args.poison_path,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    adv_noise = AdvPerturbation(args.adv_noise_num, args.image_size, save_path='{}/{}/'.format(args.save_path, args.job_id))

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    def forward_loss(net, batch, extra):
        labels = extra["y"].to(dist_util.dev())

        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        logits = net(batch, timesteps=t)
        loss = F.cross_entropy(logits, labels, reduction="none").mean()

        return loss

    last_train_image_epoch = 0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        cnt_samples = (step + resume_step + 1) * args.batch_size * dist.get_world_size()
        logger.logkv(
            "samples",
            cnt_samples,
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step, '{}/{}/'.format(args.save_path, args.job_id))

        train_image_epoch = cnt_samples // args.train_image_interval_samples
        if train_image_epoch > last_train_image_epoch:
            assert train_image_epoch == last_train_image_epoch + 1

            # for param in model.parameters():
            #     param.requires_grad = False

            for _train_perturb_step, (batch_data, batch_dict) in tqdm(enumerate(train_noise_dataloader)):
                batch_idx = batch_dict['idx']
                batch_label = batch_dict['y']
                batch_data = batch_data.to(dist_util.dev())
                batch_adv_noise = adv_noise.get_adv_noise(batch_idx)
                x_adv = (batch_data.detach() + batch_adv_noise.detach()).float()
                loop_bar = tqdm(range(args.adv_step))
                for _ in loop_bar:
                    loop_bar.set_description("Batch [{}/{}]".format(_train_perturb_step, len(train_noise_dataloader)))
                    x_adv.requires_grad_()
                    accumulated_grad = 0
                    with th.enable_grad():
                        loss = 0
                        for _ in range(args.adv_t_num):
                            loss += forward_loss(model, x_adv, batch_dict)
                    grad = th.autograd.grad(loss, [x_adv])[0]
                    accumulated_grad += grad
                    x_adv = x_adv.detach() - args.adv_alpha * th.sign(accumulated_grad.detach())
                    x_adv = th.min(th.max(x_adv, batch_data - args.adv_epsilon), batch_data + args.adv_epsilon)
                    x_adv = th.clamp(x_adv, -1.0, 1.0)

                new_adv_noise = x_adv.detach() - batch_data.detach()

                adv_noise.set_adv_noise(batch_idx, new_adv_noise)

            # for param in model.parameters():
            #     param.requires_grad = True

            adv_noise.save_adv_noise()

        last_train_image_epoch = train_image_epoch

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step, '{}/{}/'.format(args.save_path, args.job_id))
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, save_path):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(save_path, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(save_path, f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=64,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        image_size = 32,
        poisoned=False,
        poison_path='',
        random_padding_crop=False,
        job_id = "local",
        save_path='./results',
        train_image_interval_samples = 50000,
        adv_noise_num = 50000,
        adv_step=20,
        adv_epsilon=0.0628,
        adv_alpha=0.00628,
        adv_t_num=3,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
