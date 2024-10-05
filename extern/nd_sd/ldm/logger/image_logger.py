import argparse
import csv
import datetime
import glob
import importlib
import os
import pdb
import sys
import time
from functools import partial

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from einops import rearrange
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch import autocast
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from utils.color_transfer import (identity, split_rgbd_only_tensor,
                                  split_rgbd_tensor)


class ImageLogger(Callback):

    def __init__(self,
                 batch_frequency,
                 max_images,
                 epoch_frequency=10,
                 clamp=True,
                 increase_log_steps=True,
                 rescale=True,
                 disabled=False,
                 log_on_batch_idx=True,
                 log_first_step=False,
                 log_images_kwargs=None,
                 free_scale=7.5,
                 sub_name='',
                 solver='plms'):

        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.epoch_freq = epoch_frequency
        self.max_images = max_images
        self.solver = solver

        try:
            # pl_verison 1.4.2
            self.logger_log_images = {
                pl.loggers.TestTubeLogger: self._testtube,
            }
        except:
            # pl_verison 1.9
            self.logger_log_images = {
                pl.loggers.TensorBoardLogger: self._tensorboard,
            }

        times = int(np.log2(self.batch_freq * epoch_frequency))
        self.log_steps = [
            2**times - 2**(n) for n in range(
                int(np.log2(self.batch_freq * epoch_frequency)) + 1)
        ]
        self.log_steps = self.log_steps[::-1]

        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        self.to_img = {}
        self.no_cat = []
        self.free_scale = free_scale
        self.sub_name = sub_name

    def iter_times(self, batch_idx):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        cur_epoch = batch_idx // (self.batch_freq)
        check_idx = batch_idx % (self.batch_freq * self.epoch_freq)

        return check_idx, cur_epoch

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, 'images', split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)

            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        check_idx, cur_epoch = self.iter_times(batch_idx)

        if (self.check_frequency(check_idx, cur_epoch)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k].float(), -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx, cur_epoch):

        if ((cur_epoch % self.epoch_freq == 0)
                and (check_idx in self.log_steps)):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        if not self.disabled and (pl_module.global_step > 0
                                  or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):

        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split='val')
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm
                    and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def log_all_gpu(self, save_dir, split, images, global_step, current_epoch,
                    batch_idx, batch):
        root = os.path.join(save_dir, 'images', split)

        file_paths = batch['file_path_']

        k = 'samples'
        images = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

        for file_path, image in zip(file_paths, images):
            rgb = image[:3]

            rgb = rgb.permute(1, 2, 0)
            rgb = rgb.numpy()
            rgb = (rgb * 255).astype(np.uint8)

            rgb_save_name = 'rgb_' + file_path

            rgb_path = os.path.join(root, rgb_save_name)

            os.makedirs(os.path.split(rgb_path)[0], exist_ok=True)

            Image.fromarray(rgb).save(rgb_path)

    def log_test_img(self,
                     pl_module,
                     batch,
                     batch_idx,
                     split='test',
                     free_scale=7.5):
        logger = type(pl_module.logger)
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        with torch.no_grad():
            images = pl_module.sample_imgs(
                batch, ddim_steps=50, scale=free_scale)

        for k in images:
            images[k] = images[k]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k].float(), -1., 1.)

        self.log_all_gpu(pl_module.logger.save_dir, split, images,
                         pl_module.global_step, pl_module.current_epoch,
                         batch_idx, batch)

        if is_train:
            pl_module.train()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloder_idx):
        self.log_test_img(
            pl_module,
            batch,
            batch_idx,
            split='test' + '{}'.format(self.sub_name),
            free_scale=self.free_scale)


#### rgb depth-2bits
class ImageDepthLogger(ImageLogger):

    def log_img(self, pl_module, batch, batch_idx, split='train'):

        check_idx, cur_epoch = self.iter_times(batch_idx)

        if (self.check_frequency(check_idx, cur_epoch)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_rgbd(
                    batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k].float(), -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):

        for k in images:
            image = images[k]
            image = (image + 1.) / 2.
            if k not in self.to_img:
                image = split_rgbd_tensor(image)
            else:
                image = self.to_img[k](image)

            if k not in self.no_cat:
                image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image)

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, 'images', split)

        for k in images:
            if self.rescale:
                image = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            if k not in self.to_img:
                image = split_rgbd_tensor(image)
            else:
                image = self.to_img[k](image)

            if k not in self.no_cat:
                image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image, nrow=4)

            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)

            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_all_gpu(self, save_dir, split, images, global_step, current_epoch,
                    batch_idx, batch):
        root = os.path.join(save_dir, 'images', split)

        file_paths = batch['file_path_']

        k = 'samples'
        images = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

        if k not in self.to_img:
            images = split_rgbd_tensor(images)
        else:
            images = self.to_img[k](images)

        for file_path, image in zip(file_paths, images):
            rgb = image[:3]
            depth = image[3:]

            rgb = rgb.permute(1, 2, 0)
            rgb = rgb.numpy()
            rgb = (rgb * 255).astype(np.uint8)

            depth = depth.permute(1, 2, 0)
            depth = depth.numpy()
            depth = (depth * 255).astype(np.uint8)

            rgb_save_name = 'rgb_' + file_path
            depth_save_name = 'depth_' + file_path

            rgb_path = os.path.join(root, rgb_save_name)
            depth_path = os.path.join(root, depth_save_name)

            os.makedirs(os.path.split(rgb_path)[0], exist_ok=True)

            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(depth).save(depth_path)

    def check_frequency(self, check_idx, cur_epoch):
        if ((cur_epoch % self.epoch_freq == 0)
                and (check_idx in self.log_steps)):
            return True
        return False


class ImageDepthDDIMLogger(ImageDepthLogger):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_img = {
            'conditioning': identity,
            'inputs': split_rgbd_tensor,
            'reconstruction': split_rgbd_tensor,
            'diffusion_row': split_rgbd_tensor,
        }

        self.no_cat = ['conditioning']

    def log_img(self, pl_module, batch, batch_idx, split='train'):

        check_idx, cur_epoch = self.iter_times(batch_idx)

        if (self.check_frequency(check_idx, cur_epoch)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_rgbd(
                    batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k].float(), -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):

        for k in images:
            image = images[k]
            image = (image + 1.) / 2.
            if k not in self.to_img:
                image = split_rgbd_only_tensor(image)
            else:
                image = self.to_img[k](image)

            if k not in self.no_cat:
                image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image)

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):

        for k in images:
            image = images[k]
            image = (image + 1.) / 2.
            if k not in self.to_img:
                image = split_rgbd_only_tensor(image)
            else:
                image = self.to_img[k](image)

            if k not in self.no_cat:
                image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image)

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    def log_test_img(self,
                     pl_module,
                     batch,
                     batch_idx,
                     split='test',
                     free_scale=7.5,
                     solver='plms'):
        logger = type(pl_module.logger)
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        with torch.no_grad():
            images = pl_module.sample_imgs(
                batch,
                ddim_steps=50,
                scale=free_scale,
                ddim_eta=0.,
                solver=solver)

        for k in images:
            images[k] = images[k]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k].float(), -1., 1.)

        self.log_all_gpu(pl_module.logger.save_dir, split, images,
                         pl_module.global_step, pl_module.current_epoch,
                         batch_idx, batch)

        if is_train:
            pl_module.train()


####### rgbd logger


class ImageDepthOnlyLogger(ImageDepthLogger):

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):

        for k in images:
            image = images[k]
            image = (image + 1.) / 2.
            image = split_rgbd_only_tensor(image)
            image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image)

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):

        for k in images:
            image = images[k]
            image = (image + 1.) / 2.
            image = split_rgbd_only_tensor(image)
            image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)

            grid = torchvision.utils.make_grid(image)

            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, 'images', split)

        for k in images:
            if self.rescale:
                image = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            image = split_rgbd_only_tensor(image)

            image = torch.cat([image[:, :3], image[:, 3:]], dim=-1)
            grid = torchvision.utils.make_grid(image, nrow=4)

            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)

            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_all_gpu(self, save_dir, split, images, global_step, current_epoch,
                    batch_idx, batch):
        root = os.path.join(save_dir, 'images', split)

        file_paths = batch['file_path_']

        k = 'samples'
        images = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

        for file_path, image in zip(file_paths, images):
            rgb = image[:3]
            depth = image[3:]

            rgb = rgb.permute(1, 2, 0)
            rgb = rgb.numpy()
            rgb = (rgb * 255).astype(np.uint8)

            depth = depth.permute(1, 2, 0)
            depth = depth.numpy()
            depth = (depth * 255).astype(np.uint8)[..., 0]

            rgb_save_name = 'rgb_' + file_path
            depth_save_name = 'depth_' + file_path

            rgb_path = os.path.join(root, rgb_save_name)
            depth_path = os.path.join(root, depth_save_name)

            os.makedirs(os.path.split(rgb_path)[0], exist_ok=True)

            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(depth).save(depth_path)


class ImageDepthOnlyDDIMLogger(ImageDepthDDIMLogger):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_img = {
            'conditioning': identity,
            'inputs': split_rgbd_only_tensor,
            'reconstruction': split_rgbd_only_tensor,
            'diffusion_row': split_rgbd_only_tensor,
            'samples': split_rgbd_only_tensor,
        }

        self.no_cat = ['conditioning']


class ImageDepthOnlyVAELogger(ImageDepthOnlyLogger):

    def log_test_img(self,
                     pl_module,
                     batch,
                     batch_idx,
                     split='test',
                     free_scale=7.5):
        logger = type(pl_module.logger)
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        with torch.no_grad():
            images = pl_module.sample_imgs(batch)

        for k in images:
            images[k] = images[k]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k].float(), -1., 1.)

        self.log_all_gpu(pl_module.logger.save_dir, split, images,
                         pl_module.global_step, pl_module.current_epoch,
                         batch_idx, batch)

        if is_train:
            pl_module.train()


class ImageVAELogger(ImageLogger):

    def log_test_img(self,
                     pl_module,
                     batch,
                     batch_idx,
                     split='test',
                     free_scale=7.5):
        logger = type(pl_module.logger)
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        with torch.no_grad():
            images = pl_module.sample_imgs(batch)

        for k in images:
            images[k] = images[k]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k].float(), -1., 1.)

        self.log_all_gpu(pl_module.logger.save_dir, split, images,
                         pl_module.global_step, pl_module.current_epoch,
                         batch_idx, batch)

        if is_train:
            pl_module.train()
