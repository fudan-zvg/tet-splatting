import argparse
import os
import pdb
import time

import cv2
import numpy as np
from diffusers import (DDIMScheduler, DDPMScheduler,
                       DPMSolverMultistepScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionPipeline)
from transformers import AutoTokenizer, CLIPTextModel

NEGATIVE_PROMPTS = 'ugly, tiling'


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--path',
        default='./',
    )

    args = parser.parse_args()
    return args


opt = get_parse()
'''
tokenizer = AutoTokenizer.from_pretrained(
'stabilityai/stable-diffusion-2-1-base'   , subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base', subfolder="text_encoder",local_files_only = True
)
'''

# pipe = StableDiffusionPipeline.from_pretrained(opt.path,safety_checker=None, scheduler = KDPM2DiscreteScheduler()).to('cuda')
# pipe = StableDiffusionPipeline.from_pretrained(opt.path,safety_checker=None, scheduler = DDIMScheduler().from_config('./ema-only/scheduler/scheduler_config.json')).to('cuda')
pipe = StableDiffusionPipeline.from_pretrained(
    opt.path,
    safety_checker=None,
    scheduler=DPMSolverMultistepScheduler().from_config(
        os.path.join(opt.path, 'scheduler/scheduler_config.json'))).to('cuda')
# pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',safety_checker=None, scheduler= DDIMScheduler(), local_files_only=True).to('cuda')

# pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',safety_checker=None, local_files_only=True).to('cuda')

prompts = [
    'a close up of a sheet of pizza on a table.',
    'A picture of some lemons on a table.',
    'A little girl with a pink bow in her hair eating broccoli.',
    'A man is on a path riding a horse.',
    'A muffin in a black muffin wrap next to a fork.',
    'a white polar bear drinking water from a water source next to some rocks',
    'A baby bunny sitting on top of a stack of pancakes.',
    'A blue poison-dart frog sitting on a water lily.',
    'a peacock on a surfboard.',
    'a photo of a green apple',
    'a photo of a beautiful woman',
    'a zoomed out DSLR photo of the Sydney opera house, aerial view.',
    'A ripe strawberry.',
    'An ice cream sundae.',
    'A ladybug.',
    'A stack of pancakes covered in maple syrup.',
    'a photograph of an astronaut riding a horse.',
    'a bike is chained to a lamp post outside',
    "A motorcyclist's head is reflected as he rides in his side mirror.",
    'A DSLR photo of a car made out of sushi.',
    'A beautiful dress made out of garbage bags, on a mannequin. Studio lighting, high quality, high resolution.',
    'a bagel filled with cream cheese and lox.',
    'a plate piled high with chocolate chip cookies.',
    'the Imperial State Crown of England.',
    'A silver platter piled high with fruits.',
    'a silver candelabra sitting on a red velvet tablecloth, only one candle is lit.',
    'A squirrel wearing a leather jacket riding a motorcycle.',
    'A delicious croissant',
    "a metal sculpture of a lion's head, highly detailed",
    'A highly detailed sandcastle',
    'A fresh cinnamon roll covered in glaze, high resolution',
    'A car made out of cheese',
    'A vintage record player',
    'A highly detailed stone bust of Theodoros Kolokotronis',
]

prompts = [
    'a yellow car on a white background',
]
sub = ', high detail 3D model'
pre = 'the albedo of '
# pre = ''

for prompt in prompts:
    prompt = pre + prompt + sub

    name = '_'.join(prompt.split(' '))
    rgbd = pipe(prompt, negative_prompt=NEGATIVE_PROMPTS).images[0]

    rgbd = np.asarray(rgbd)
    if rgbd.shape[-1] == 4:
        rgb = rgbd[..., :3]
        depth = rgbd[..., 3:]
        depth = np.concatenate([depth, depth, depth], axis=-1)
        rgbd = np.concatenate([rgb, depth], axis=1)

    save_path = os.path.join('outputs/sampling_mydiffusers',
                             '{}.png'.format(name))

    cv2.imwrite(save_path, rgbd[..., ::-1])
