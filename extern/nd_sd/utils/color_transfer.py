import pdb

import numpy as np
import torch
from einops import repeat


def map_2_16bit(x):
    x = (np.clip(x, 0, 1.) * 65535).astype(np.uint16)

    low_x = np.zeros_like(x)
    low_x[x < 256] = x[x < 256]
    high_x = x >> 8

    return np.concatenate(
        [np.zeros_like(low_x[..., None]), high_x[..., None], low_x[..., None]],
        axis=-1).astype(np.uint8)


def map_16bit_2_8(x):

    x = x.astype(np.uint16)
    ret_v = x[..., 1] << 8 + x[..., 0]

    return ret_v / 65535.


def split_rgbd(x, is_bgr=False):
    '''
    x: np.uint8
    '''

    rgb, depth = x[..., :3], x[..., 3:]
    if is_bgr:
        rgb = rgb[..., ::-1]

    depth = (map_16bit_2_8(depth) * 255).astype(np.uint8)
    depth = np.repeat(depth[..., None], 3, axis=-1)
    rgbd = np.concatenate([rgb, depth], axis=1)

    return rgbd


def split_rgbd_tensor(x_tensor):

    # depth is from [0 1]
    rgb, depth = torch.split(x_tensor, 3, dim=1)
    depth = depth * 255
    depth_v = depth[:, 1] * 255 + depth[:, 0]
    depth_v = depth_v / 65535
    depth_v = repeat(depth_v[:, None], 'b 1 h w -> b 3 h w')

    return torch.cat([rgb, depth_v], dim=1)


def split_rgbd_only_tensor(x_tensor):

    # depth is from [0 1]
    rgb, depth = x_tensor[:, :3], x_tensor[:, 3]
    depth_v = repeat(depth[:, None], 'b 1 h w -> b 3 h w')

    return torch.cat([rgb, depth_v], dim=1)


def identity(x):
    return x
