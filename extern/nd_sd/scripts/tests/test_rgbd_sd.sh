#!/bin/bash

# python3 vae_main.py --base ./configs/stable-diffusion/sd_depth_mse.yaml -t --scale_lr False --gpus 0,1,2,3,4,5,6,7,
python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
