import pdb
import time

import torch
from diffusers import StableDiffusionPipeline

model_id = 'stabilityai/stable-diffusion-2'

faliure = True
while faliure:
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16)
    except:
        print('failure download')
        time.sleep(1)
        continue
    faliure = False
