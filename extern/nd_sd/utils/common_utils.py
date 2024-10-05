"""
@File: format_transfer.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-06
@Desc: common settings
"""

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


def upper_bound(arr, left, right, target):
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left


def lower_bound(arr, left, right, target):
    while left < right:
        mid = (left + right) >> 1

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


# print arguments
def print_args(args):
    print(
        '################################  args  ################################'
    )
    for k, v in args.__dict__.items():
        print('{0: <10}\t{1: <30}\t{2: <20}'.format(k, str(v), str(type(v))))
    print(
        '########################################################################'
    )


# torch.no_grad warpper for functions
def make_nograd_func(func):

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):

    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_recursive_meta_func(meta, parents=''):
    meta_infos = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            meta_infos.update(make_recursive_meta_func(v, parents + k + '/'))
    elif isinstance(meta, list) or isinstance(meta, tuple):
        # only support the length of list is lq 3
        for iter_name, v in enumerate(meta[:3]):
            meta_infos.update(
                make_recursive_meta_func(
                    v, parents + '{:03d}'.format(iter_name) + '/'))
    else:
        return {parents[:-1]: meta}
    return meta_infos


@make_recursive_func
def unsqueeze_tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars.unsqueeze(0)
    elif isinstance(vars, Garment_Polygons):
        return vars
    elif isinstance(vars, Garment_Mesh):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for unsqueeze_tensor'.format(type(vars)))


@make_recursive_func
def inverse_image_normalized(vars):
    if isinstance(vars, torch.Tensor):
        vars = (vars / 2 + 0.5) * 255
        return vars
    elif isinstance(vars, np.ndarray):
        vars = (vars / 2 + 0.5) * 255
        return vars
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for squeeze_tensor'.format(type(vars)))


@make_recursive_func
def bgr2rgb(vars):
    if isinstance(vars, np.ndarray):
        #NOTE that no consider batch
        if len(vars.shape) == 3:
            vars = vars[..., ::-1]
        return vars
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for squeeze_tensor'.format(type(vars)))


@make_recursive_func
def resize256(vars):
    if isinstance(vars, np.ndarray):
        if len(vars.shape) == 3:
            h, w, _ = vars.shape
        if len(vars.shape) == 2:
            h, w = vars.shape
        ratio = 256. / min(h, w)
        return cv2.resize(vars, (int(ratio * w), int(ratio * h)))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for squeeze_tensor'.format(type(vars)))


@make_recursive_func
def resize512(vars):
    if isinstance(vars, np.ndarray):
        if len(vars.shape) == 3:
            h, w, _ = vars.shape
        if len(vars.shape) == 2:
            h, w = vars.shape
        ratio = 512. / min(h, w)
        return cv2.resize(vars, (int(ratio * w), int(ratio * h)))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for squeeze_tensor'.format(type(vars)))


@make_recursive_func
def squeeze_tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars.squeeze(0)
    elif isinstance(vars, Garment_Polygons):
        return vars
    elif isinstance(vars, Garment_Mesh):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for squeeze_tensor'.format(type(vars)))


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError(
            'invalid input type {} for tensor2float'.format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError(
            'invalid input type {} for tensor2numpy'.format(type(vars)))


@make_recursive_func
def numpy2tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars
    elif isinstance(vars, np.ndarray):
        return torch.from_numpy(vars)
    else:
        raise NotImplementedError(
            'invalid input type {} for numpy2tensor'.format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device('cuda'))
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, Meshes):
        return vars.to(torch.device('cuda'))
    else:
        raise NotImplementedError('invalid input type {} for tocuda'.format(
            type(vars)))


@make_recursive_func
def tocpu(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device('cpu'))
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, Meshes):
        return vars.to(torch.device('cpu'))
    else:
        raise NotImplementedError('invalid input type {} for tocpu'.format(
            type(vars)))
