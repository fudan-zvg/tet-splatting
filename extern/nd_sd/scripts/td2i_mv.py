import argparse
import os
import random
import sys
sys.path.append('./')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ldm.camera_utils import get_camera
from ldm.data.objaverse_camera import Objaverse_Dataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from model_zoo import build_model
from omegaconf import OmegaConf
from PIL import Image



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_img(imgs):

    if imgs.shape[-1] == 4:
        rgb = imgs[..., :3]
        depth = imgs[..., 3:]
        depth = np.concatenate([depth, depth, depth], -1)

        imgs = np.concatenate([rgb, depth], axis=0)

    return imgs


def t2i(model,
        image_size,
        prompt,
        uc,
        sampler,
        step=20,
        scale=7.5,
        batch_size=8,
        ddim_eta=0.,
        dtype=torch.float32,
        device='cuda',
        camera=None,
        num_frames=1,
        as_latents=False,
        cond_method='ori',
        depth=None,
        normal=None):

    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad():
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {'context': c.repeat(batch_size, 1, 1)}
        uc_ = {'context': uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_['camera'] = uc_['camera'] = camera
            c_['num_frames'] = uc_['num_frames'] = num_frames

        if cond_method == 'ori':
            pass
        elif cond_method == 'cat_n':
            normal_z = torch.nn.functional.interpolate(
                normal,
                size=(image_size // 8, image_size // 8),
                mode='nearest')

            c_['c_concat'] = normal_z
            uc_['c_concat'] = normal_z

        elif cond_method == 'cat_d':
            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode='nearest')

            c_['c_concat'] = depth_z
            uc_['c_concat'] = depth_z

        elif cond_method == 'cat_nd':

            normal_z = torch.nn.functional.interpolate(
                normal,
                size=(image_size // 8, image_size // 8),
                mode='nearest')

            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode='nearest')

            nd_z = torch.cat([normal_z, depth_z], dim=1)
            c_['c_concat'] = nd_z
            uc_['c_concat'] = nd_z

        else:
            raise NotImplementedError

        shape = [4, image_size // 8, image_size // 8]

        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None)

        if not as_latents:
            x_sample = model.decode_first_stage(samples_ddim)
        else:
            x_sample = F.interpolate(
                samples_ddim, (image_size, image_size), mode='bilinear')

        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str, default='albedo-mv', help='model name')
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='load model from local config (override model_name)')
    parser.add_argument(
        '--ckpt_path', type=str, default=None, help='path to local checkpoint')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=', 3d asset')
    parser.add_argument('--pre', type=str, default='the albedo of ')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument(
        '--num_frames',
        type=int,
        default=4,
        help='num of frames (views) to generate')
    parser.add_argument('--use_camera', type=int, default=1)
    parser.add_argument('--as_latents', type=int, default=0)
    parser.add_argument('--camera_elev', type=int, default=15)
    parser.add_argument('--camera_azim', type=int, default=45)
    parser.add_argument('--camera_azim_span', type=int, default=360)
    parser.add_argument('--camera_dist', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--depth_file', type=str, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.depth_file), args.depth_file
    assert args.prompt is not None

    depth_np = cv2.imread(args.depth_file)

    if 'wovae' in args.model_name:
        args.as_latents = 1
    print(vars(args))

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print('load t2i model ... ')
    if args.config_path is None:
        model, config = build_model(
            args.model_name, ckpt_path=args.ckpt_path, return_cfg=True)
    else:
        assert args.ckpt_path is not None, 'ckpt_path must be specified!'
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))

    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning(['']).to(device)
    print('load t2i model done . ')

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    prompt = args.prompt

    show_name = os.path.join(save_dir, '_'.join(prompt.split(' ')) + '.png')
    depth = torch.from_numpy(depth_np).float().permute(2, 0, 1).to(device)
    depth = depth / 127.5 - 1.0
    depth = depth.reshape(3, 256, 4, 256)
    depth = depth.permute(2, 0, 1, 3)[:, 0:1]
    print(depth.min(), depth.max(), depth.shape)

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(
            args.num_frames,
            elevation=args.camera_elev,
            azimuth_start=args.camera_azim,
            azimuth_span=args.camera_azim_span,
            camera_distance=args.camera_dist
        )  # note that camera distance is very important
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
    else:
        camera = None

    t = args.pre + prompt + args.suffix
    print('prompt: {}!'.format(t))
    set_seed(args.seed)
    images = []

    for j in range(5):
        img = t2i(
            model,
            args.size,
            t,
            uc,
            sampler,
            step=50,
            scale=args.scale,
            batch_size=batch_size,
            ddim_eta=0.0,
            dtype=dtype,
            device=device,
            camera=camera,
            num_frames=args.num_frames,
            as_latents=args.as_latents,
            cond_method=config.model.params.cond_method,
            depth=depth,
            normal=None)

        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)

    cv2.imwrite(
        show_name,
        np.concatenate([depth_np, np.asarray(show_img(images))],
                       axis=0)[:, :, (2, 1, 0)])
