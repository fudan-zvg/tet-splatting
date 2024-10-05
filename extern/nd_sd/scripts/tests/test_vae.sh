#!/bin/bash

# python3 vae_main.py --base ./configs/stable-diffusion/sd_depth_mse.yaml -t --scale_lr False --gpus 0,1,2,3,4,5,6,7,
# python3 vae_main.py --base ./configs/autoencoder_rgbd/autoencoder_kl_rgbd_1e-5.yaml --resume ./logs/vae_sd_14/checkpoints/last.pth --scale_lr False  --gpus 0,1,2,3,4,5,6,7,
# python3 vae_main.py --base ./configs/autoencoder_rgbd/autoencoder_kl_rgbd_1e-5.yaml --resume ./logs/2023-05-28T16-39-28_autoencoder_kl_rgbd_1e-5/checkpoints/last.ckpt --scale_lr False  --gpus 0,1,2,3,4,5,6,7,
# python3 vae_main.py --base ./configs/autoencoder_rgbd/autoencoder_kl_rgbd_1e-5.yaml --resume ./logs/vae_sd_21/checkpoints/last.pth --scale_lr False  --gpus 0,1,2,3,4,5,6,7,
python3 vae_main.py --base ./configs/autoencoder_rgbd/autoencoder_kl_rgbd_1e-5_laion2b.yaml --resume ./logs/2023-06-07T13-36-54_autoencoder_kl_rgbd_1e-5_laion2b/checkpoints/last.ckpt --scale_lr False  --gpus 0,
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgb_1e-5.yaml --resume ./logs/sd_rgb_1e-5/checkpoints/last.ckpt --scale_lr False  --gpus 0,
