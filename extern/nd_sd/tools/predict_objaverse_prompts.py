import glob
import os
import pdb
import sys
sys.path.append('./')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import json
import lavis
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.read_exr import (blender2midas, read_camera_matrix, read_depth,
                            read_exr, read_exr_to_normal, read_w2c)



normal_paths = sorted(
    glob.glob('./objaverse/294277b770a342d9a75486837c7c72da/*_normal.exr'))
depth_paths = sorted(
    glob.glob('./objaverse/294277b770a342d9a75486837c7c72da/*_depth.exr'))
camera_path = './objaverse/294277b770a342d9a75486837c7c72da/transforms.json'
w2c_cameras = read_camera_matrix(camera_path)

for cnt, normal_path in enumerate(normal_paths):
    normal = torch.from_numpy(read_exr(normal_path)[0]).float()
    camera = w2c_cameras[cnt]
    w2c, cam_dis = read_w2c(camera)
    dis = read_depth(depth_paths[cnt], cam_dis)

    R = torch.from_numpy(w2c[:3, :3]).float()
    pdb.set_trace()
    im = torch.einsum('rc,hwc->hwr', R, normal)
    im = blender2midas(im)

    im = (im + 1) / 2
    im = im.clamp(0, 1.)
    im = im.detach().cpu().squeeze().numpy()

    plt.imsave(os.path.join('./debug/vis_normal/{:03d}.png'.format(cnt)), im)

    dis = (dis + 1) / 2
    dis = dis.clamp(0, 1.)
    dis = dis.detach().cpu().squeeze().numpy()

    plt.imsave(os.path.join('./debug/vis_depth/{:03d}.png'.format(cnt)), dis)
