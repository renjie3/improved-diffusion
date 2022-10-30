import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from tqdm import tqdm

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def _list_model_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "model" in entry and ext.lower() in ["pt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class AdvLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_path='./results/',
        adv_noise=None,
        adv_step=20,
        adv_epsilon=0.0628,
        adv_alpha=0.00628,
        target_image=None,
        adv_loss_type="mse_attack_noisefunction",
        group_model=False,
        group_model_list=None,
        random_noise_every_adv_step=False,
        eot_gaussian_num=1,
        t_seg_num=8,
        t_seg_start=3,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.save_path = save_path

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.adv_noise = adv_noise
        if adv_noise is not None:
            self.one_class_image_num = adv_noise.shape[0]
        else:
            self.one_class_image_num = 1
        self.adv_step=adv_step
        self.adv_epsilon=adv_epsilon
        self.adv_alpha=adv_alpha
        self.single_target_image = target_image
        self.adv_loss_type = adv_loss_type
        self.group_model = group_model
        self.group_model_list = group_model_list
        self.random_noise_every_adv_step = random_noise_every_adv_step
        self.eot_gaussian_num = eot_gaussian_num
        self.t_seg_num = t_seg_num
        self.t_seg_start = t_seg_start

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            # print(dist_util.dev())
            # input('check dist_util.dev()')
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            if "y" in cond.keys():
                micro_cond = {
                    "y": cond["y"][i : i + self.microbatch].to(dist_util.dev())
                }
            else:
                micro_cond = {}
            last_batch = (i + self.microbatch) >= batch.shape[0] # the ending microbatch, not the microbatch before current one.
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # print(i, losses["loss"].shape)

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def run_adv(self):
        if not os.path.exists(self.get_blob_logdir()):
            os.mkdir(self.get_blob_logdir())
        self.ddp_model.eval()
        # set the model parameters to be fixed
        for param in self.ddp_model.parameters():
            param.requires_grad = False

        if self.group_model:
            for k in range(len(self.group_model_list)):
                self.group_model_list[k].eval()
                # set the model parameters to be fixed
                for param in self.group_model_list[k].parameters():
                    param.requires_grad = False

        for _idx, (batch, cond) in enumerate(self.data):
            logger.log("Batch id {}".format(_idx))
            # print('check 1')
            # print(cond)
            # continue

            batch_classes = cond['output_classes']
            batch_idx = cond['idx']
            HIDDEN_CLASS = 0
            hidden_idx_in_batch = batch_classes == HIDDEN_CLASS # select all the samples that belongs to birds
            batch = batch[hidden_idx_in_batch]
            batch_idx = batch_idx[hidden_idx_in_batch]
            x_natural = batch.to(dist_util.dev())

            if batch_idx.shape[0] == 0:
                continue
            
            batch_adv_noise = self._get_adv_noise(batch_idx)

            # print('check 2')

            target_image = th.tensor(self.single_target_image).unsqueeze(0).repeat([batch_adv_noise.shape[0], 1, 1, 1]).to(dist_util.dev())

            all_t_list = []
            all_weights_list = []
            t_seg_num = self.t_seg_num
            t_seg_start = self.t_seg_start
            t_range_len = 1. / float(t_seg_num)
            for i_t in range(t_seg_num):
                t, weights = self.schedule_sampler.range_sample(x_natural.shape[0], dist_util.dev(), start=i_t*t_range_len, end=(i_t+1)*t_range_len)
                all_t_list.append(t)
                all_weights_list.append(weights)

            eot_gaussian_num = self.eot_gaussian_num
            all_gaussian_noise = th.randn([eot_gaussian_num*t_seg_num, *x_natural.shape]).to(dist_util.dev())

            x_adv = (x_natural.detach() + batch_adv_noise.detach()).float()
            adv_step_loop_bar = tqdm(range(self.adv_step))
            for _ in adv_step_loop_bar:
                adv_step_loop_bar.set_description("Batch [{}/{}]".format(_idx, len(self.data) // 4))
                x_adv.requires_grad_()
                accumulated_grad = 0
                for i in range(t_seg_start, t_seg_num):
                    # print('t_seg_num: ', i)
                    t = all_t_list[i]
                    weights = all_weights_list[i]
                    for j in range(eot_gaussian_num):
                        # print('eot_gaussian_num: ', j)
                        gaussian_noise = all_gaussian_noise[i*eot_gaussian_num + j]
                        with th.enable_grad():
                            if not self.group_model:
                                loss = self.adv_loss(x_adv, cond, adv_loss_type=self.adv_loss_type, target_image=target_image, gaussian_noise=gaussian_noise, t=t, weights=weights)
                                grad = th.autograd.grad(loss, [x_adv])[0]
                                accumulated_grad += grad
                            else:
                                for k in range(len(self.group_model_list)):
                                    if self.random_noise_every_adv_step:
                                        gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                        # input('check wrong')
                                    # print(len(self.group_model_list), k)
                                    loss = self.adv_loss(x_adv, cond, adv_loss_type=self.adv_loss_type, target_image=target_image, gaussian_noise=gaussian_noise, t=t, weights=weights, group_idx=k)
                                    grad = th.autograd.grad(loss, [x_adv])[0]
                                    # print("grad", grad[0,0,0])
                                    # print("loss", loss)
                                    # input('check')
                                    accumulated_grad += grad
                        
                x_adv = x_adv.detach() - self.adv_alpha * th.sign(accumulated_grad.detach()) # TODO pay attention to whether it should be positive or negative gradient direction.
                x_adv = th.min(th.max(x_adv, x_natural - self.adv_epsilon), x_natural + self.adv_epsilon)
                x_adv = th.clamp(x_adv, -1.0, 1.0)

            new_adv_noise = x_adv.detach() - x_natural.detach()

            self._set_adv_noise(batch_idx, new_adv_noise)
            
            # print(cond)
            # input('check')

            # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            # for _ in range(perturb_steps):
            #     x_adv.requires_grad_()
            #     with torch.enable_grad():
            #         loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
            #     grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            #     x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            #     x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            #     x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # with open('test.npy', 'wb') as f:
            # np.save(f, np.array([1, 2]))

        self._save_adv_noise()

    def adv_loss(self, batch, cond, adv_loss_type="mse_attack_noisefunction", target_image=None, gaussian_noise=None, t=None, weights=None, group_idx=0,):
        if adv_loss_type == "mse_attack_noisefunction":
            if gaussian_noise is None:
                raise('gaussian_noise is None.')
            if t is None:
                raise('t is None.')
            if weights is None:
                raise('weigths is None.')
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                if "y" in cond.keys():
                    micro_cond = {
                        "y": cond["y"][i : i + self.microbatch].to(dist_util.dev())
                    }
                else:
                    micro_cond = {}
                last_batch = (i + self.microbatch) >= batch.shape[0] # the ending microbatch, not the microbatch before current one.
                # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                if self.group_model:
                    compute_losses = functools.partial(
                        self.diffusion.mse_attack_noisefunction,
                        self.group_model_list[group_idx],
                        micro,
                        t,
                        model_kwargs=micro_cond,
                        noise=gaussian_noise,
                        target_image=target_image,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.mse_attack_noisefunction,
                        self.ddp_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                        noise=gaussian_noise,
                        target_image=target_image,
                    )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                # else:
                #     with self.ddp_model.no_sync():
                #         losses = compute_losses()

                # if isinstance(self.schedule_sampler, LossAwareSampler):
                #     self.schedule_sampler.update_with_local_losses(
                #         t, losses["loss"].detach()
                #     )

                loss = (losses * weights).mean()
                
                return loss

        elif adv_loss_type == "forward_bachword_loss":
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                if "y" in cond.keys():
                    micro_cond = {
                        "y": cond["y"][i : i + self.microbatch].to(dist_util.dev())
                    }
                else:
                    micro_cond = {}
                last_batch = (i + self.microbatch) >= batch.shape[0] # the ending microbatch, not the microbatch before current one.
                # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                if self.group_model:
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.group_model_list[group_idx],
                        micro,
                        t,
                        noise=gaussian_noise,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.ddp_model,
                        micro,
                        t,
                        noise=gaussian_noise,
                        model_kwargs=micro_cond,
                    )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                
                return loss

        elif adv_loss_type == "negative_forward_bachword_loss":
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                if "y" in cond.keys():
                    micro_cond = {
                        "y": cond["y"][i : i + self.microbatch].to(dist_util.dev())
                    }
                else:
                    micro_cond = {}
                last_batch = (i + self.microbatch) >= batch.shape[0] # the ending microbatch, not the microbatch before current one.
                # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                if self.group_model:
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.group_model_list[group_idx],
                        micro,
                        t,
                        noise=gaussian_noise,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.ddp_model,
                        micro,
                        t,
                        noise=gaussian_noise,
                        model_kwargs=micro_cond,
                    )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = - (losses["loss"] * weights).mean()
                
                return loss

    def _get_adv_noise(self, idx):
        adv_noise_numpy = self.adv_noise[idx]
        return th.tensor(adv_noise_numpy).to(dist_util.dev())

    def _set_adv_noise(self, idx, batch_noise):
        self.adv_noise[idx] = batch_noise.cpu().numpy()
        # return th.tensor(adv_noise_numpy).to(dist_util.dev())

    def _save_adv_noise(self, ):
        # os.mkdir(self.get_blob_logdir())
        with open(bf.join(self.get_blob_logdir(), "adv_noise.npy"), "wb") as f:
            np.save(f, self.adv_noise)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        print(self.get_blob_logdir())

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    def get_blob_logdir(self):
        # return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())
        return self.save_path


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
