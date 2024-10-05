#!/bin/bash
# for debug:
# --gpus 0, data.params.batch_size=2 model.params.unet_config.params.model_channels=64 model.params.ckpt_path=null

unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

python3 main.py --base ./configs/stable-diffusion/normald/sd_1_5/txt_cond/objaverse/txtcond_mvsd-4-objaverse_finetune_wovae.yaml -t --scale_lr False \
    --gpus 0,1,2,3,4,5,6,7 ${@:1}
