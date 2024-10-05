#!/bin/bash
# python3 vae_main.py --base ./configs/autoencoder_rgbd/autoencoder_kl_rgbd_1e-5_laion2b.yaml -t --scale_lr False --gpus 0,1,2,3,4,5,6,7,
python3 main.py --base ./configs/autoencoder_normal_depth/autoencoder_normal_depth.yaml -t --scale_lr False  ${@:1}
