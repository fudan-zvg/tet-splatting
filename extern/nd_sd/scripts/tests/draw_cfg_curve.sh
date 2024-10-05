
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
RESUME=$1
CFGLIST=(1. 2. 3. 4. 5. 6. 7. 8. 9. 10.)
config='./configs/stable-diffusion/rgbd/sd_1_5/rgbd_laion2b_cfg_inference.yaml'

for CFG in ${CFGLIST[@]}
    do
    python3 draw_ddim_cfg_curve.py --base $config  --scale_lr False --cfg $CFG --resume $RESUME --gpus 0,1,2,3,4,5,6,7,
done
