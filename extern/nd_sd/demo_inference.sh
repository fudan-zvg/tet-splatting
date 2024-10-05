#!/bin/bash

# nd
# export ckpt_path=/path/to/nd-ckpt
export ckpt_path="./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/nd-laion_ema.ckpt"
export prompt="a living room with a sofa"
export save_dir="outputs/nd"
# dmp solver
python ./scripts/t2i.py --ckpt $ckpt_path --prompt "$prompt" --dpm_solver --n_samples 2 --save_dir $save_dir


# nd-mv
# export ckpt_path=/path/to/nd-mv-ckpt
export ckpt_path="./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/nd_mv_ema.ckpt"
export prompt="a cute girl"
export save_dir="outputs/nd-mv"
python ./scripts/t2i_mv.py --ckpt_path $ckpt_path --prompt "$prompt"  --num_frames 4  --model_name nd-mv --save_dir $save_dir


# albedo-mv with depth map condition
# export ckpt_path=/path/to/albedo-mv-ckpt
export ckpt_path="./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/albedo_mv_ema.ckpt"
export prompt="a wooden chair"
export save_dir="outputs/albedo-mv"
export depth_file="./loads/chair_depth.png"
python ./scripts/td2i_mv.py --ckpt_path $ckpt_path --prompt "$prompt" --depth_file $depth_file --num_frames 4  --model_name albedo-mv --save_dir $save_dir
