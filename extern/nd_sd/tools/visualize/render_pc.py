import argparse
import glob
import os
import pdb
import sys

import matplotlib.pyplot as plt
# Util function for loading point clouds|
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from pytorch3d.renderer import (AlphaCompositor, FoVOrthographicCameras,
                                NormWeightedCompositor,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, PulsarPointsRenderer,
                                look_at_view_transform)
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import (AxisArgs, plot_batch_individually,
                                      plot_scene)

need_pytorch3d = False

try:
    import pytorch3d
except ModuleNotFoundError:
    xxxx


def png2video(save_path):
    img_path = os.path.join(save_path, '*.png')
    video_path = os.path.join(save_path, 'visualize.mp4')
    if os.path.exists(video_path):
        os.remove(video_path)
    cmd = 'ffmpeg -r {} -pattern_type glob -i \'{}\' -vcodec libx264 -crf 18 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {}'.format(
        10, img_path, video_path)
    os.system(cmd)


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pc_path', default='', help='train or test')
    parser.add_argument('--save_path', default='cuda', help='select model')
    parser.add_argument('--ranges', default=30, help='select model')

    args = parser.parse_args()
    return args


def camera_path(dist=20, elev=10, ranges=30):
    R, T = look_at_view_transform([40 for i in range(ranges)],
                                  [10 for i in range(ranges)],
                                  torch.arange(-60, 60, 120 / ranges))

    return R, T


def normalized(verts, scale_ratio=1.):
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    verts += (-center)
    verts *= (1.0 / float(scale))

    return verts * scale_ratio


if __name__ == '__main__':
    args = get_parse()
    ranges = args.ranges
    pc_files = glob.glob(os.path.join(args.pc_path, '*.ply'))

    for pc_file in pc_files:

        pc = trimesh.load(pc_file, vertex_colors=True)
        verts = pc.vertices  # (V, 3)
        colors = pc.visual.vertex_colors  # (V, 4)
        colors = colors[:, :3] / 255.0

        save_path = os.path.join(args.save_path,
                                 pc_file.split('/')[-1].replace('.ply', ''))
        os.makedirs(save_path, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')

        verts = torch.Tensor(verts).to(device)
        rgb = torch.Tensor(colors).to(device)
        verts = normalized(verts, scale_ratio=0.8)
        verts[..., -1] = -verts[..., -1]
        verts[..., -2] = -verts[..., -2]

        point_cloud = Pointclouds(
            points=[verts for i in range(ranges)],
            features=[rgb for i in range(ranges)])

        R, T = camera_path(ranges=ranges)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.005,
        )

        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0, 0, 0)))
        with torch.no_grad():
            images = renderer(point_cloud)

        ## save folder
        for i in range(images.shape[0]):
            save_item = os.path.join(save_path, '{:05d}.png'.format(i))
            rgb = (images[i, ..., :3].detach().cpu().numpy() * 255).astype(
                np.uint8)
            im = Image.fromarray(rgb).save(save_item)
            print(save_item)

        png2video(save_path)
