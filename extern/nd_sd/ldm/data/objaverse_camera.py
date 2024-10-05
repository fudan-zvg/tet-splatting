'''
given cap3d caption, lingteng qiu render objaverse
rendering albedo models;
v6 is four view image w/o albedo check, depth_normalized, background (0,0,1)
'''
import bisect
import glob
import multiprocessing
import os
import pdb
import pickle
import random
import shutil
import sys
sys.path.append('./')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import tarfile
from functools import partial
from io import BytesIO

import albumentations
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import pandas as pd
import PIL
import pytorch_lightning as pl
import taming.data.utils as tdu
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as TF
import webdataset as wds
import yaml
from einops import rearrange
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.modules.image_degradation import (degradation_fn_bsr,
                                           degradation_fn_bsr_light)
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from taming.data.imagenet import (ImagePaths, download,
                                  give_synsets_from_indices, retrieve,
                                  str_to_indices)
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              random_split)
from torchvision import transforms
from tqdm import tqdm
from utils.color_transfer import map_2_16bit, map_16bit_2_8, split_rgbd
from utils.common_utils import numpy2tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[..., 0] = img[..., 0]
    img[..., -1] = -img[..., -1]
    return img


def disparity_normalized(depth, camera_distance):

    near = 0.866  #sqrt(3) * 0.5
    far = camera_distance + 0.866
    near_distance = camera_distance - near

    near_disparity = 1. / near_distance
    far_disparity = 1. / far

    disparity = 1. / depth[:, 0, ...]
    disparity[disparity <= far_disparity] = far_disparity

    disparity = (disparity - far_disparity) / (near_disparity - far_disparity)

    disparity = disparity * 2 - 1
    disparity = torch.clamp(disparity, -1, 1)

    return disparity


def depth_normalized(depth, camera_distance):

    # near = 0.866 #sqrt(3) * 0.5
    far_distance = camera_distance + 0.866
    near_distance = camera_distance - 0.866

    depth = depth[:, 0, ...]
    depth[depth > far_distance] = far_distance
    depth[depth < near_distance] = far_distance  # out of box

    depth = (far_distance - depth) / (far_distance - near_distance)

    depth = depth * 2 - 1
    depth = torch.clamp(depth, -1, 1)

    return depth


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    '''
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    '''

    return camera_matrix


def lerp(a, b, mask):
    return (1 - mask) * a + mask * b


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id
                                               * split_size:(worker_id + 1)
                                               * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Objaverse_Dataset(Dataset):

    def __init__(self,
                 json_path=None,
                 caption_path=None,
                 data_root=None,
                 size=None,
                 degradation=None,
                 downscale_f=4,
                 min_crop_f=0.8,
                 max_crop_f=1.,
                 random_crop=False,
                 debug: bool = False,
                 views=24,
                 validation=False,
                 folder_key='campos_512_v4',
                 as_video=True,
                 pre_str='',
                 suff_str=', 3d asset',
                 albedo_check=False,
                 filter_box: bool = True,
                 color_key='_albedo',
                 select_view=[0, 6, 12, 18],
                 fix_view=False):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        :param as_video: as video containers?
        :param pre_str: the pre string of caption?
        :param views: rendering views

        """
        assert json_path is not None
        self.items = self.read_json(json_path)
        self.data_root = data_root
        self.color_key = color_key
        self.objaverse_key = sorted(self.items)

        with open(caption_path, 'r') as reader:
            self.folder2caption = json.load(reader)

        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA)
        self.pil_interpolation = False  # gets reset later if incase interp_op is from pillow
        self.views = views
        self.validation = validation
        self.folder_key = folder_key

        self.debug = debug
        self.as_video = as_video
        self.pre_str = pre_str
        self.suff_str = suff_str
        self.albedo_check = albedo_check
        self.filter_box = filter_box
        self.select_view = select_view
        self.fix_view = fix_view

        if degradation == 'bsrgan':
            self.degradation_process = partial(
                degradation_fn_bsr, sf=downscale_f)

        elif degradation == 'bsrgan_light':
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
                'cv_nearest': cv2.INTER_NEAREST,
                'cv_bilinear': cv2.INTER_LINEAR,
                'cv_bicubic': cv2.INTER_CUBIC,
                'cv_area': cv2.INTER_AREA,
                'cv_lanczos': cv2.INTER_LANCZOS4,
                'pil_nearest': PIL.Image.NEAREST,
                'pil_bilinear': PIL.Image.BILINEAR,
                'pil_bicubic': PIL.Image.BICUBIC,
                'pil_box': PIL.Image.BOX,
                'pil_hamming': PIL.Image.HAMMING,
                'pil_lanczos': PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith('pil_')

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize,
                    size=self.LR_size,
                    interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)

    def read_json(self, json_file):
        with open(json_file, 'r') as reader:
            return json.load(reader)

    def __len__(self):
        # for validation only 100 items
        return len(self.objaverse_key) if not self.validation else 100

    def valid(self, data):
        return len(
            data['caption']) > 0 and data['normal'].shape[-1] == 3 and data[
                'depth'].shape[-1] == 1 and data['albedo'].shape[-1] == 3

    def retrival(self, name, index_cond, background=[0, 0, 1]):
        ''' condition is normal map
        target is normal + disparity
        '''

        # 16-bits depth
        # our blender render it from [0,5]

        def read_dnormal(idx, cond_pos):
            cond_cam_dis = np.linalg.norm(cond_pos, 2)

            near = 0.867  #sqrt(3) * 0.5
            near_distance = cond_cam_dis - near

            normald_path = os.path.join(
                img_folder, '{:05d}/{:05d}_nd.exr'.format(idx, idx))
            normald = cv2.imread(normald_path,
                                 cv2.IMREAD_UNCHANGED).astype(np.float32)
            # 0,1
            normal = normald[..., :3]
            # unity2blender
            normal = normal[..., [2, 0, 1]]
            depth = normald[..., 3:]
            depth[depth < near_distance] = 0

            #-1,1
            normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims=True))
            # depth has some problems

            normal = normal / normal_norm
            normal = np.nan_to_num(normal, nan=-1.)

            normal_mask = depth == 0.

            return normal, depth, normal_mask

        img_folder = os.path.join(self.data_root, name, self.folder_key)
        camera_path = os.path.join(
            img_folder, '{:05d}/{:05d}.json'.format(index_cond, index_cond))

        cond_c2w = read_camera_matrix_single(camera_path)
        camera_embedding = torch.from_numpy(cond_c2w.flatten().astype(
            np.float32))

        cond_pos = cond_c2w[:3, 3:]
        world_normal, world_depth, normal_mask = read_dnormal(
            index_cond, cond_pos)

        world_normal = torch.from_numpy(world_normal)
        view_cn = world_normal
        view_cn = blender2midas(world_normal @ (cond_c2w[:3, :3])).float()
        view_cn[normal_mask[..., 0]] = torch.Tensor(background).float()

        world_depth = torch.from_numpy(world_depth)
        world_depth[world_depth == 0] = 5.
        cond_cam_dis = np.linalg.norm(cond_pos, 2)

        cond_disparity = depth_normalized(world_depth[None, None, ..., 0],
                                          cond_cam_dis)[0, ..., None]

        albedo_path = os.path.join(
            img_folder, '{:05d}/{:05d}'.format(index_cond, index_cond)
            + self.color_key + '.png')
        albedo = cv2.imread(albedo_path)[..., [2, 1, 0]]
        albedo = albedo.astype(np.float32) / 255.
        albedo[normal_mask[..., 0]] = [1, 1, 1]

        albedo = (albedo * 2 - 1)
        albedo = torch.from_numpy(albedo).float()

        return view_cn, cond_disparity, albedo, camera_embedding

    def debug_img(self, img, path='debug', name='example.png'):

        img = rearrange(img, 'b h w c -> h (b w) c')

        img = (img + 1) / 2
        img = (img.detach().cpu() * 255).numpy().astype(np.uint8)

        save_name = os.path.join(path, name)

        if img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = img[..., [2, 1, 0]]

        cv2.imwrite(save_name, img)

    def contain_orthogonal_view(self,
                                objaverse_name,
                                idx_cond,
                                target_group=(2, 2)):

        def merge_group(img_container):
            # [C, h*views, w]
            img = torch.cat(img_container, dim=1)
            h, w, c = img.shape
            w_bins = w // target_group[-1]
            imgs = torch.split(img, w_bins, dim=1)
            assert len(imgs) == target_group[0]
            imgs = torch.cat(imgs, dim=0)
            return imgs

        normal_list = []
        disparity_list = []
        albedo_list = []
        camera_embedding_list = []

        orthogonal_view = self.select_view  # orthogonal space: 0. 90. 180. 270.

        for view in orthogonal_view:
            view = (idx_cond + view) % self.views
            normal, disparity, albedo, camera_embedding = self.retrival(
                objaverse_name, view)
            normal_list.append(normal)
            disparity_list.append(disparity)
            albedo_list.append(albedo)
            camera_embedding_list.append(camera_embedding)

        if not self.as_video:
            normal = merge_group(normal_list)
            disparity = merge_group(disparity_list)
            albedo = merge_group(albedo_list)
            camera_embedding = torch.cat(camera_embedding_list, dim=0)

        else:
            # as video sd
            normal = torch.cat([normal[None] for normal in normal_list], dim=0)
            disparity = torch.cat(
                [disparity[None] for disparity in disparity_list], dim=0)
            albedo = torch.cat([albedo[None] for albedo in albedo_list], dim=0)
            camera_embedding = torch.cat([
                camera_embedding[None]
                for camera_embedding in camera_embedding_list
            ],
                                         dim=0)

            normal = F.interpolate(
                normal.permute(0, 3, 1, 2), (self.size, self.size),
                mode='nearest').permute(0, 2, 3, 1)
            disparity = F.interpolate(
                disparity.permute(0, 3, 1, 2), (self.size, self.size),
                mode='nearest').permute(0, 2, 3, 1)
            albedo = F.interpolate(
                albedo.permute(0, 3, 1, 2), (self.size, self.size),
                mode='nearest').permute(0, 2, 3, 1)

        return normal, disparity, albedo, camera_embedding

    def isnot_nan(self, ret_dict):
        normal = ret_dict['normal']
        depth = ret_dict['depth']
        albedo = ret_dict['albedo']

        return torch.isnan(normal).sum() == 0 and torch.isnan(
            depth).sum() == 0 and torch.isnan(albedo).sum() == 0

    def __getitem__(self, item):

        ret_dict = {}

        while True:
            try:
                objaverse_name = self.objaverse_key[item]
                caption = self.folder2caption[objaverse_name]

                if self.filter_box:
                    if 'box' in caption or 'cube' in caption:
                        raise ValueError('containing cube or box in caption')

                index_cond = random.sample(range(self.views),
                                           1)[0] if not self.fix_view else 0

                normal, disparity, albedo, camera_embedding = self.contain_orthogonal_view(
                    objaverse_name, index_cond)

                ret_dict['normal'] = numpy2tensor(normal).float()
                ret_dict['depth'] = numpy2tensor(disparity).float()
                ret_dict['albedo'] = numpy2tensor(albedo).float()
                ret_dict['caption'] = caption
                ret_dict['item'] = item
                ret_dict['camera'] = camera_embedding
                ret_dict['image'] = torch.cat(
                    [ret_dict['normal'], ret_dict['depth']], dim=-1)

                assert (self.valid(ret_dict))
                assert (self.isnot_nan(ret_dict))

                if self.albedo_check:
                    assert not (albedo == 1).sum() / albedo.numel(
                    ) > 0.95  # all white about 10 % data is all white

                ret_dict[
                    'caption'] = self.pre_str + caption[:-1] + self.suff_str

                if self.debug:
                    self.debug_img(
                        albedo, name='albedo_{:04d}.png'.format(item))
                    self.debug_img(
                        normal, name='normal_{:04d}.png'.format(item))
                    self.debug_img(
                        disparity, name='disparity_{:04d}.png'.format(item))
                break

            except:
                item = (item + 1) % len(self)

        return ret_dict

    # using to debug
    def visualize(self, img, normal, depth):

        def to_01(img):
            return np.clip((img + 1) / 2, 0., 1.)

        img = to_01(img)
        normal = to_01(normal)
        depth = to_01(depth)
        plt.imsave('./debug/vis_objaverse/image.png', img)
        plt.imsave('./debug/vis_objaverse/depth.png', depth)
        plt.imsave('./debug/vis_objaverse/normal.png', normal)


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):

        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else min(
            batch_size * 2, multiprocessing.cpu_count())

        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs['predict'] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def setup(self, stage=None):
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = False
        init_fn = worker_init_fn

        train_loader = DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=init_fn)

        return train_loader

    def _val_dataloader(self, shuffle=False):

        if isinstance(self.datasets['validation'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn)


if __name__ == '__main__':
    #20824 laion2b_5_33,2 = wds.WebDataset('../improved_aesthetics_5plus/laion-2ben-5_1/{00000..20824}.tar')
    obj_dataset = Objaverse_Dataset(
        '/mnt/cap_objaverse/dataset/raw/0/valid_paths_v4_cap_filter_thres_28.json',
        size=256,
        degradation='cv_bilinear',
        views=24,
        debug=True,
        pre_str='the albedo of ',
        folder_key='campos_512_v4')

    times = 0

    black_item_caption = []

    for i in range(0, len(obj_dataset)):
        # normal [512,512,3]
        data = obj_dataset[i]

        print(data['caption'])
        print(data['item'])

    with open('black_item.json', 'w') as writer:
        json.dump(black_item_caption, writer)
    xxxx

    train_loader = DataLoader(
        obj_dataset,
        batch_size=8,
        num_workers=16,
        shuffle=False,
        worker_init_fn=worker_init_fn)

    for item in train_loader:
        pdb.set_trace()
