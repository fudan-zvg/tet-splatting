''' file to
'''
import pdb

import cv2
import json
import numpy as np
import torch


def read_exr(exr):
    # bgr2rgb
    im = cv2.imread(exr, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    return im[..., ::-1].copy()


def read_exr_to_normal(exr):
    im = cv2.imread(exr, cv2.IMREAD_UNCHANGED)[None]
    # bgr 2 rgb
    im = im[..., ::-1].copy()

    norm = (np.linalg.norm(im, 2, axis=-1, keepdims=True))
    mask = (norm[..., 0] == 0)
    im = im / norm
    im = np.nan_to_num(im, nan=-1.)
    im[mask] = [-1, -1, -1]

    # mask using to produce background
    return im, mask


def read_camera_matrix(camera):
    with open(camera, 'r') as reader:
        cameras = json.load(reader)
        camera_frames = cameras['frames']

    return camera_frames


def read_w2c(camera):
    tm = camera['transform_matrix']
    tm = np.asarray(tm)

    cam_pos = tm[:3, 3:]

    world2cam = np.zeros_like(tm)
    world2cam[:3, :3] = tm[:3, :3].transpose()
    world2cam[:3, 3:] = -tm[:3, :3].transpose() @ tm[:3, 3:]
    world2cam[-1, -1] = 1

    return world2cam, np.linalg.norm(cam_pos, 2, axis=0)


def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''

    img[..., 0] = -img[..., 0]
    return img


#


def read_depth(exr, camera_distance=3):
    '''bbox [-0.5, 0.5]
    '''

    depth = cv2.imread(exr, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    near = 0.866  #sqrt(3) * 0.5
    far = camera_distance + 0.866
    near_distance = camera_distance - near
    near_disparity = 1. / near_distance
    far_disparity = 1. / far

    disparity = 1. / depth[..., 0]
    disparity[disparity <= far_disparity] = far_disparity
    disparity = (disparity - far_disparity) / (near_disparity - far_disparity)

    disparity = disparity * 2 - 1
    disparity = np.clip(disparity, -1, 1)

    return disparity
