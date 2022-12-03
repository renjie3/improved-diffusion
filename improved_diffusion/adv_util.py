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
    zero_grad_no_detach,
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
            results.extend(_list_model_files_recursively(full_path))
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
        poison_mode="gradient_matching",
        source_data_loader=None,
        source_clean_loader=None,
        optim_mode="adam",
        tau=0.1,
        t_seg_end=4,
        sample_t_gaussian_in_loop=True,
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
        self.differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
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
        self.t_seg_end = t_seg_end
        self.poison_mode = poison_mode
        self.source_data_loader = source_data_loader
        self.source_clean_loader = source_clean_loader
        self.optim_mode = optim_mode
        self.tau = tau
        self.sample_t_gaussian_in_loop = sample_t_gaussian_in_loop

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

        if self.poison_mode != "gradient_matching":
            # input("cehck gradient_matching")
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
            t_seg_end = self.t_seg_end
            t_range_len = 1. / float(t_seg_num)
            if self.adv_loss_type != "test_t_emb_emb_loss":
                for i_t in range(t_seg_num):
                    t, weights = self.schedule_sampler.range_sample(x_natural.shape[0], dist_util.dev(), start=i_t*t_range_len, end=(i_t+1)*t_range_len)
                    all_t_list.append(t)
                    all_weights_list.append(weights)
            else:
                for i_t in range(t_seg_num):
                    t, weights = self.schedule_sampler.range_sample(x_natural.shape[0], dist_util.dev(), start=0.1, end=0.11)
                    all_t_list.append(t)
                    all_weights_list.append(weights)

            eot_gaussian_num = self.eot_gaussian_num
            all_gaussian_noise = th.randn([eot_gaussian_num*t_seg_num, *x_natural.shape]).to(dist_util.dev())

            x_adv = (x_natural.detach() + batch_adv_noise.detach()).float()
            adv_step_loop_bar = tqdm(range(self.adv_step))
            for _ in adv_step_loop_bar:
                adv_step_loop_bar.set_description("Batch [{}/{}]".format(_idx, len(self.data) // 2)) # // 2 because we have two class, bird and horse and we only train adv on bird.
                x_adv.requires_grad_()
                accumulated_grad = 0
                accumulated_loss = 0
                for i in range(t_seg_start, t_seg_end):
                    # print('t_seg_num: ', i)
                    t = all_t_list[i]
                    weights = all_weights_list[i]
                    for j in range(eot_gaussian_num):
                        # print('t_seg_num: ', i, 'eot_gaussian_num: ', j)
                        gaussian_noise = all_gaussian_noise[i*eot_gaussian_num + j]
                        with th.enable_grad():
                            if not self.group_model:
                                if self.random_noise_every_adv_step:
                                    gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                # print(x_natural.shape)
                                loss = self.adv_loss(x_adv, cond, x_natural=x_natural, adv_loss_type=self.adv_loss_type, target_image=target_image, gaussian_noise=gaussian_noise, t=t, weights=weights)
                                grad = th.autograd.grad(loss, [x_adv])[0]
                                accumulated_grad += grad
                                # print("loss", loss)
                                # input('check')
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
                        # print(loss.item())
                        accumulated_loss += loss.item()
                print("accumulated_loss:", accumulated_loss)
                        
                x_adv = x_adv.detach() - self.adv_alpha * th.sign(accumulated_grad.detach())
                x_adv = th.min(th.max(x_adv, x_natural - self.adv_epsilon), x_natural + self.adv_epsilon)
                x_adv = th.clamp(x_adv, -1.0, 1.0)

            input('check')

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

    def adv_loss(self, batch, cond, x_natural, adv_loss_type="mse_attack_noisefunction", target_image=None, gaussian_noise=None, t=None, weights=None, group_idx=0,):
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

        elif adv_loss_type == "test_t_emb_emb_loss":
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
                        self.diffusion.test_t_emb_emb_loss,
                        self.group_model_list[group_idx],
                        micro,
                        t,
                        x_0=x_natural,
                        model_kwargs=micro_cond,
                        noise=gaussian_noise,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.test_t_emb_emb_loss,
                        self.ddp_model,
                        micro,
                        t,
                        x_0=x_natural,
                        model_kwargs=micro_cond,
                        noise=gaussian_noise,
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

    def run_adv_gm(self):
        if not os.path.exists(self.get_blob_logdir()):
            if dist_util.dev() == th.device("cuda:0"):
                os.mkdir(self.get_blob_logdir())
        self.ddp_model.eval()
        if self.poison_mode != "gradient_matching":
            # set the model parameters to be fixed
            for param in self.ddp_model.parameters():
                param.requires_grad = False

            if self.group_model:
                for k in range(len(self.group_model_list)):
                    self.group_model_list[k].eval()
                    # set the model parameters to be fixed
                    for param in self.group_model_list[k].parameters():
                        param.requires_grad = False

        for _idx, (batch_data, source_batch_data, source_clean_batch_data) in enumerate(zip(self.data, self.source_data_loader, self.source_clean_loader)):
            logger.log("Batch id {}".format(_idx))

            (batch, cond) = batch_data
            (source_batch, source_cond) = source_batch_data
            (source_clean_batch, source_clean_cond) = source_clean_batch_data

            _batch_size = batch.shape[0]
            _source_batch_size = source_batch.shape[0]
            _mapping_radio = _batch_size // _source_batch_size
            
            source_batch = source_batch.to(dist_util.dev())
            source_clean_batch = source_clean_batch.to(dist_util.dev())

            batch_classes = cond['output_classes']
            batch_idx = cond['idx']
            x_natural = batch.to(dist_util.dev())

            source_batch_idx = source_cond['idx']
            source_clean_batch_idx = source_clean_cond['idx']

            # print(batch_idx)
            # print(source_batch_idx)
            # print(source_clean_batch_idx)

            # input("checkcheck")

            
            batch_adv_noise = self._get_adv_noise(batch_idx)

            all_t_list = []
            all_weights_list = []
            t_seg_num = self.t_seg_num
            t_seg_start = self.t_seg_start
            t_seg_end = self.t_seg_end
            t_range_len = 1. / float(t_seg_num)

            if not self.sample_t_gaussian_in_loop:
                for i_t in range(t_seg_num):
                    t, weights = self.schedule_sampler.range_sample(source_clean_batch.shape[0], dist_util.dev(), start=i_t*t_range_len, end=(i_t+1)*t_range_len)
                    all_t_list.append(t)
                    all_weights_list.append(weights)
                eot_gaussian_num = self.eot_gaussian_num
                all_gaussian_noise = th.randn([eot_gaussian_num*t_seg_num, *source_clean_batch.shape]).to(dist_util.dev())

            x_adv = (x_natural.detach() + batch_adv_noise.detach()).float()

            if self.optim_mode == "adam":
                x_adv.requires_grad_()
                att_optimizer = th.optim.Adam([x_adv], lr=self.tau, weight_decay=0)

            adv_step_loop_bar = tqdm(range(self.adv_step))
            for _ in adv_step_loop_bar:
                if self.sample_t_gaussian_in_loop:
                    for i_t in range(t_seg_num):
                        t, weights = self.schedule_sampler.range_sample(source_clean_batch.shape[0], dist_util.dev(), start=i_t*t_range_len, end=(i_t+1)*t_range_len)
                        all_t_list.append(t)
                        all_weights_list.append(weights)
                    eot_gaussian_num = self.eot_gaussian_num
                    all_gaussian_noise = th.randn([eot_gaussian_num*t_seg_num, *source_clean_batch.shape]).to(dist_util.dev())

                adv_step_loop_bar.set_description("Batch [{}/{}]".format(_idx, len(self.data)))
                
                accumulated_source_grad = 0
                accumulated_source_count = 0
                accumulated_poison_grad = 0
                accumulated_poison_count = 0
                accumulated_loss = 0
                accumulated_count = 0
                for i in range(t_seg_start, t_seg_end):
                    # print('t_seg_num: ', i)
                    source_t = all_t_list[i]
                    # print(source_t)
                    poison_t = source_t.unsqueeze(1).repeat([1,_mapping_radio]).reshape([-1])
                    source_weights = all_weights_list[i]
                    poison_weights = source_weights.unsqueeze(1).repeat([1,_mapping_radio]).reshape([-1])

                    for j in range(eot_gaussian_num):
                        if self.optim_mode != "adam":
                            x_adv.requires_grad_()
                        print('t_seg_num: ', i, 'eot_gaussian_num: ', j)
                        source_gaussian_noise = all_gaussian_noise[i*eot_gaussian_num + j]
                        poison_gaussian_noise = source_gaussian_noise.unsqueeze(1).repeat([1, _mapping_radio, 1, 1, 1]).reshape([-1, 3, 32, 32])
                        with th.enable_grad():
                            # -------------- get source gradient
                            if not self.group_model:
                                if self.random_noise_every_adv_step:
                                    gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                    raise("random_noise_every_adv_step should not be used here or tobe developed.")
                                source_grad = self._compute_source_gradients(source_batch, source_clean_batch, gaussian_noise=source_gaussian_noise, t=source_t, weights=source_weights)
                                # accumulated_source_grad += source_grad
                                # accumulated_source_count += 1
                            else:
                                raise("Group_model Not emplemented.")
                                for k in range(len(self.group_model_list)):
                                    if self.random_noise_every_adv_step:
                                        gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                        raise("random_noise_every_adv_step should not be used here or tobe developed.")
                                    source_grad = self._compute_source_gradients(source_batch, source_clean_batch, gaussian_noise=source_gaussian_noise, t=source_t, weights=source_weights, group_idx=k)
                                    # accumulated_source_grad += source_grad
                                    # accumulated_source_count += 1
                            # -------------- get poison gradient
                            if not self.group_model:
                                if self.random_noise_every_adv_step:
                                    gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                    raise("random_noise_every_adv_step should not be used here or tobe developed.")
                                poison_grad = self._compute_poison_gradients(x_adv, gaussian_noise=poison_gaussian_noise, t=poison_t, weights=poison_weights)
                                # accumulated_poison_grad += poison_grad
                                # accumulated_poison_count += 1
                            else:
                                for k in range(len(self.group_model_list)):
                                    if self.random_noise_every_adv_step:
                                        gaussian_noise = th.randn([*x_natural.shape]).to(dist_util.dev())
                                        raise("random_noise_every_adv_step should not be used here or tobe developed.")
                                    poison_grad = self._compute_poison_gradients(x_adv, gaussian_noise=poison_gaussian_noise, t=poison_t, weights=poison_weights, group_idx=k)
                                    # accumulated_poison_grad += poison_grad
                                    # accumulated_poison_count += 1
                        # print(loss.item())
                
                        one_adv_step_loss = self.gradient_matching_loss(source_grad, poison_grad)
                        one_adv_step_loss.backward()
                        accumulated_loss += one_adv_step_loss.item()
                        accumulated_count += 1
                        # print(one_adv_step_loss.item())
                        # one_adv_step_loss = th.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                        if self.optim_mode == "pgd":
                            x_adv = x_adv.detach() - self.adv_alpha * th.sign(x_adv.grad.detach())
                        elif self.optim_mode == "adam":
                            att_optimizer.step()
                            att_optimizer.zero_grad()
                        x_adv.data = th.min(th.max(x_adv.data, x_natural - self.adv_epsilon), x_natural + self.adv_epsilon)
                        x_adv.data = th.clamp(x_adv.data, -1.0, 1.0)
                print(accumulated_loss / accumulated_count / float(len(source_grad)))
                # print(th.min(x_adv.detach() - x_natural.detach()))
                # print(th.max(x_adv.detach() - x_natural.detach()))
            # print(dist_util.dev())

            new_adv_noise = x_adv.detach() - x_natural.detach()

            if dist_util.dev() == th.device("cuda:0"):

                gpu_batch_idx = batch_idx.to(dist_util.dev())

                batch_idx_list = [th.zeros_like(gpu_batch_idx) for _ in range(dist_util.device_num())]
                new_adv_noise_list= [th.zeros_like(new_adv_noise) for _ in range(dist_util.device_num())]
                dist.gather(new_adv_noise, new_adv_noise_list)
                dist.gather(gpu_batch_idx, batch_idx_list)

                for tosave_batch_id, tosave_new_adv_noise in zip(batch_idx_list, new_adv_noise_list):
                    # print(dist_util.dev(), tosave_batch_id)
                    # input("check")
                    self._set_adv_noise(tosave_batch_id.cpu().detach(), tosave_new_adv_noise)
                self._save_adv_noise()

            else:
                
                gpu_batch_idx = batch_idx.to(dist_util.dev())
                dist.gather(new_adv_noise)
                dist.gather(gpu_batch_idx)

    def _mix_source_and_source_clean(self, source, source_clean, t):
        T = self.diffusion.num_timesteps
        alpha_source = - 8 / float(T) * t + 4
        alpha_source = th.clip(alpha_source, min=0, max=1).reshape([-1, 1, 1, 1]).to(source.device)
        # print(alpha_source.shape)
        # print(source.shape)
        # print(source_clean.shape)
        mix_source = alpha_source * source + (1-alpha_source) * source_clean
        return mix_source.detach()

    def _compute_source_gradients(self, source, source_clean=None, gaussian_noise=None, t=None, weights=None, group_idx=0,):
        zero_grad_no_detach(self.model_params)
        input_mix_source = self._mix_source_and_source_clean(source, source_clean, t)
        output_mix_source = self._mix_source_and_source_clean(source, source_clean, t-1)
        micro_cond = {}
        
        if self.group_model:
            compute_losses = functools.partial(
                self.diffusion.training_losses_for_gm,
                self.group_model_list[group_idx],
                input_mix_source,
                t,
                output_mix_source,
                noise=gaussian_noise,
                model_kwargs=micro_cond,
            )
        else:
            compute_losses = functools.partial(
                self.diffusion.training_losses_for_gm,
                self.ddp_model,
                input_mix_source,
                t,
                output_mix_source,
                noise=gaussian_noise,
                model_kwargs=micro_cond,
            )
        losses = compute_losses()
        source_loss = (losses["loss"] * weights).mean()
        # print(source_loss.item())
        # input("check source_loss")
        grad = th.autograd.grad(source_loss, self.differentiable_params, only_inputs=True)
        # for g in grad:
        #     print(th.sum(g))
        #     input("check _compute_source_gradients")
        # for p in self.differentiable_params:
        #     print(p.requires_grad)
        #     input("check")
        return grad

    def _compute_poison_gradients(self, x_poison, gaussian_noise=None, t=None, weights=None, group_idx=0,):
        zero_grad_no_detach(self.model_params)
        micro_cond = {}
        
        if self.group_model:
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.group_model_list[group_idx],
                x_poison,
                t,
                noise=gaussian_noise,
                model_kwargs=micro_cond,
            )
        else:
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                x_poison,
                t,
                noise=gaussian_noise,
                model_kwargs=micro_cond,
            )
        losses = compute_losses()
        poison_loss = (losses["loss"] * weights).mean()
        grad = th.autograd.grad(poison_loss, self.differentiable_params, retain_graph=True, create_graph=True)
        # for g in grad:
        #     print(th.sum(g))
        #     input("check")
        return grad

    def gradient_matching_loss(self, source_grad, poison_grad, adv_loss_type="gradient_matching"):
        if adv_loss_type == "gradient_matching":

            indices = th.arange(len(source_grad))
            gm_loss = 0
            debug_sum0_count = 0

            for i in indices:
                # print(th.sum(poison_grad[i]))
                # if th.sum(poison_grad[i]) == 0:
                #     debug_sum0_count += 1
                # else:
                #     print(poison_grad[i])
                gm_loss -= th.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)

            # print("{}/{}".format(debug_sum0_count, len(indices)))

            # print("gm_loss.item():", gm_loss.item() / float(len(source_grad)))
            
            return gm_loss

    def _get_adv_noise(self, idx):
        adv_noise_numpy = self.adv_noise[idx]
        return th.tensor(adv_noise_numpy).to(dist_util.dev())

    def _set_adv_noise(self, idx, batch_noise):
        self.adv_noise[idx] = batch_noise.cpu().numpy()
        # return th.tensor(adv_noise_numpy).to(dist_util.dev())

    def _save_adv_noise(self, ):
        # os.mkdir(self.get_blob_logdir())
        adv_noise_save_path = bf.join(self.get_blob_logdir(), "adv_noise.npy")
        print("The adv noise is saved at {}.".format(adv_noise_save_path))
        with open(adv_noise_save_path, "wb") as f:
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
