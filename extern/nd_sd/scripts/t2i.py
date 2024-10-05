import argparse
import glob
import os
import pdb
import sys
sys.path.append('./')
import time
from contextlib import contextmanager, nullcontext
from itertools import islice

import cv2
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from model_zoo import build_model
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import AutoFeatureExtractor
from utils.color_transfer import (map_2_16bit, map_16bit_2_8, split_rgbd,
                                  split_rgbd_only_tensor, split_rgbd_tensor)


# load safety model
'''
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
'''

NEGATIVE_PROMPTS = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft.'


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open('assets/rick.jpeg').convert('RGB').resize(
            (hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    return x_image, False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='the prompt to render')
    parser.add_argument(
        '--save_dir',
        type=str,
        nargs='?',
        help='dir to write results to',
        default='outputs/txt2img-samples')
    parser.add_argument(
        '--skip_grid',
        action='store_true',
        help=
        'do not save a grid, only individual samples. Helpful when evaluating lots of samples',
    )
    parser.add_argument(
        '--skip_save',
        action='store_true',
        help='do not save individual samples. For speed measurements.',
    )
    parser.add_argument(
        '--ddim_steps',
        type=int,
        default=50,
        help='number of ddim sampling steps',
    )
    parser.add_argument(
        '--plms',
        action='store_true',
        help='use plms sampling',
    )
    parser.add_argument(
        '--dpm_solver',
        action='store_true',
        help='use dpm_solver sampling',
    )
    parser.add_argument(
        '--laion400m',
        action='store_true',
        help='uses the LAION400M model',
    )
    parser.add_argument(
        '--fixed_code',
        action='store_true',
        help='if enabled, uses the same starting code across samples ',
    )
    parser.add_argument(
        '--ddim_eta',
        type=float,
        default=0.0,
        help='ddim eta (eta=0.0 corresponds to deterministic sampling',
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1,
        help='sample this often',
    )
    parser.add_argument(
        '--H',
        type=int,
        default=512,
        help='image height, in pixel space',
    )
    parser.add_argument(
        '--W',
        type=int,
        default=512,
        help='image width, in pixel space',
    )
    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )
    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help=
        'how many samples to produce for each given prompt. A.k.a. batch size',
    )
    parser.add_argument(
        '--n_rows',
        type=int,
        default=0,
        help='rows in the grid (default: n_samples)',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help=
        'unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )
    parser.add_argument(
        '--from-file',
        type=str,
        help='if specified, load prompts from this file',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/inference/nd/nd-1.5-inference.yaml',
        help='path to config which constructs model',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='models/ldm/txt2depth/last.ckpt',
        help='path to checkpoint of model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the seed (for reproducible sampling)',
    )
    parser.add_argument(
        '--precision',
        type=str,
        help='evaluate at this precision',
        choices=['full', 'autocast'],
        default='autocast')
    opt = parser.parse_args()

    seed_everything(opt.seed)

    ckpt_name = os.path.splitext(os.path.basename(opt.ckpt))[0]

    outdir = os.path.join(opt.save_dir, ckpt_name)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    # config = OmegaConf.load(f"{opt.config}")
    # model = load_model_from_config(config, f"{opt.ckpt}")
    model = build_model('nd', opt.ckpt, strict=False)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    print(
        'Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...'
    )
    wm = 'StableDiffusionV1'
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f'reading prompts from {opt.from_file}')
        with open(opt.from_file, 'r') as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    if opt.prompt is None:
        prompts = [
            'a close up of a sheet of pizza on a table.',
            'A picture of some lemons on a table.',
            'A little girl with a pink bow in her hair eating broccoli.',
            'A highly detailed stone bust of Theodoros Kolokotronis',
        ]
    else:
        prompts = [opt.prompt]

    sub = ''

    for prompt_id, prompt in enumerate(prompts):
        if prompt[-1] == '.':
            prompt = prompt[:-1]
        data = [batch_size * [prompt + sub]]

        save_path = os.path.join(outpath, 'normal-depth')
        os.makedirs(save_path, exist_ok=True)
        base_count = prompt_id
        grid_count = prompt_id

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn(
                [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f],
                device=device)

        precision_scope = autocast if opt.precision == 'autocast' else nullcontext
        with torch.no_grad():
            with precision_scope('cuda'):
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc='Sampling'):
                    for prompts in tqdm(data, desc='data'):
                        uc = None
                        if opt.scale != 1.0:
                            # uc = model.get_learned_conditioning(batch_size * [""])
                            uc = model.get_learned_conditioning(
                                batch_size * [NEGATIVE_PROMPTS])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        samples_ddim, _ = sampler.sample(
                            S=50,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code)
                        '''
                        rgbd= (samples_ddim[0]).permute(1,2,0)
                        rgb = rgbd[...,:3]
                        depth = rgbd[...,3]
                        cv2.imwrite('rgb.png',((rgb.clamp(-1,1).detach().cpu().numpy()+1)[...,::-1] /2 * 255).astype(np.uint8))
                        cv2.imwrite('depth.png', ((depth.clamp(-1,1).detach().cpu().numpy()+1) /2 * 255).astype(np.uint8))
                        '''

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = split_rgbd_only_tensor(x_samples_ddim)
                        x_samples_ddim = x_samples_ddim.cpu().permute(
                            0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(
                            x_samples_ddim)
                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image).permute(0, 3, 1, 2)
                        x_checked_image_torch = torch.cat([
                            x_checked_image_torch[:, :3],
                            x_checked_image_torch[:, 3:]
                        ],
                                                          dim=-1)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(
                                    x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(
                            grid, 'c h w -> h w c').cpu().numpy()
                        h, w, c = grid.shape
                        grid = np.concatenate(
                            [grid[:, :w // 2, :], grid[:, w // 2:, :]], axis=0)

                        img = Image.fromarray(grid.astype(np.uint8))
                        img = put_watermark(img, None)

                        img.save(
                            os.path.join(save_path,
                                         f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

    print(f'Your samples are ready and waiting for you here: \n{outpath} \n'
          f' \nEnjoy.')


if __name__ == '__main__':
    main()
