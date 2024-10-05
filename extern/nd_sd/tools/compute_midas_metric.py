import argparse
import glob
# -*- coding: utf-8 -*-
import os
import os.path
import pdb
import sys
from pathlib import Path
sys.path.append('./')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import tqdm
from libs.omnidata_torch.data.transforms import get_transform
from libs.omnidata_torch.lib.midas import MidasDetector
from libs.omnidata_torch.lib.utils import HWC3, resize_image
from PIL import Image
from torchvision import transforms



def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=6000, type=int)

    args = parser.parse_args()
    return args


args = get_parse()

trans_topil = transforms.ToPILImage()
map_location = (lambda storage, loc: storage.cuda()
                ) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = args.path

img_files = sorted(glob.glob(os.path.join(path, 'rgb_*')))


def iter_laion(start, end):
    for i in range(start, end):
        item = data_container[i].replace('.jpg', '')
        item_data = int(item) // 10000
        yield item, os.path.join(
            data_path.format(item_data), data_container[i])


image_size = 384

model = MidasDetector()

trans_rgb = transforms.Compose([
    transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
    transforms.CenterCrop(512)
])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value
                               * len(sorted_img)):int((1 - trunc_value)
                                                      * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, save_depth, cnt):
    with torch.no_grad():
        save_img = os.path.join('./outputs/midas_views',
                                img_path.split('/')[-1])

        if os.path.exists(save_depth):
            print('exiting!')
            return

        print(f'Reading input {img_path} ...')
        img = np.asarray(Image.open(img_path))

        if len(img.shape) < 3:
            img = np.repeat(img[..., None], 3, axis=-1)
        h, w, c = img.shape
        if max(h, w) / min(h, w) > 4:
            print('error, {}'.format(img_path))
            return

        img = resize_image(HWC3(img), image_size)
        depth_img, depth_tensor = model(img)
        depth_img = HWC3(depth_img)
        np.savez(
            save_depth, values=depth_tensor.detach().cpu().float().numpy())
        if cnt % 100 == 0:
            img = np.asarray(Image.open(img_path))
            sd_depth_img = np.asarray(
                Image.open(img_path.replace('rgb_COCO', 'depth_COCO')))
            depth_img = cv2.resize(depth_img, (512, 512))
            merge_img = np.concatenate(
                [img[..., [2, 1, 0]], sd_depth_img, depth_img], axis=1)

            cv2.imwrite(save_img, merge_img)


cnt = 0
for data in tqdm.tqdm(img_files[args.start:args.end]):
    # for data in  zip(data_names, img_files):
    img_path = data
    save_path = data.replace('rgb_COCO', 'midas_COCO').replace('.jpg', '.npz')

    save_outputs(img_path, save_path, cnt)
    cnt += 1
