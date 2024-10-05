import argparse
import os
import random
import sys
sys.path.append('./')

import numpy as np
import torch
import torch.nn.functional as F
from ldm.camera_utils import get_camera
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
        tokens = []
        rgb_tensors = imgs[..., :3]
        depth_tensors = imgs[..., 3:]
        depth_tensors = np.concatenate(
            [depth_tensors, depth_tensors, depth_tensors], -1)

        batch_size = rgb_tensors.shape[0]

        for rgb, depth in zip(
                np.split(rgb_tensors, batch_size // 4, axis=0),
                np.split(depth_tensors, batch_size // 4, axis=0)):
            tokens.append(rgb)
            tokens.append(depth)

        imgs = np.concatenate(tokens, axis=0)

        ret_imgs = []
        for i in range(0, imgs.shape[0] // 4):
            cur_imgs = imgs[i * 4:(i + 1) * 4]
            cur_img_list = np.split(cur_imgs, 4)
            cur_imgs = np.concatenate(cur_img_list, axis=2)[0]
            ret_imgs.append(cur_imgs)

        imgs = np.concatenate(ret_imgs, axis=0)

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
        as_latents=False):

    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad():
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {'context': c.repeat(batch_size, 1, 1)}
        uc_ = {'context': uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_['camera'] = uc_['camera'] = camera
            c_['num_frames'] = uc_['num_frames'] = num_frames

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
        '--model_name',
        type=str,
        default='nd-mv',
        choices=['nd-mv', 'nd-mv-vae'],
        help='load pre-trained model from hugginface')
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='prompt txt file path or prompt')
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='load model from local config (override model_name)')
    parser.add_argument(
        '--ckpt_path', type=str, default=None, help='path to local checkpoint')
    parser.add_argument('--suffix', type=str, default=', 3d asset')
    parser.add_argument('--pre', type=str, default='')
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
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    if args.model_name == 'nd-mv':
        args.as_latents = 1
    print(vars(args))

    prompts = []
    if os.path.isfile(args.prompt):
        with open(args.prompt, 'r') as reader:
            for line in reader:
                prompt = line.strip()
                prompts.append(prompt)
    else:
        prompts.append(args.prompt)

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print('load t2i model ... ')
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
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

    for prompt in prompts:
        print(prompt)

        # pre-compute camera matrices
        if args.use_camera:
            camera = get_camera(
                args.num_frames,
                elevation=args.camera_elev,
                azimuth_start=args.camera_azim,
                azimuth_span=args.camera_azim_span,
                camera_distance=2.0
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
                scale=10,
                batch_size=batch_size,
                ddim_eta=0.0,
                dtype=dtype,
                device=device,
                camera=camera,
                num_frames=args.num_frames,
                as_latents=args.as_latents)

            images.append(img)
        images = np.concatenate(images, 0)

        show_name = os.path.join(save_dir,
                                 '_'.join(prompt.split(' ')) + '.png')

        Image.fromarray(show_img(images)).save(show_name)
