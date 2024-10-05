#!/bin/bash

# python3 vae_main.py --base ./configs/stable-diffusion/sd_depth_mse.yaml -t --scale_lr False --gpus 0,1,2,3,4,5,6,7,
python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgb_2_1.yaml --resume ./logs/sd_rgb_2_1/checkpoints/last.pth --scale_lr False  --gpus 0,1,2,3,4,5,6,7,
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgb_1e-5.yaml --resume ./logs/sd_rgb_1e-5/checkpoints/last.ckpt --scale_lr False  --gpus 0,
