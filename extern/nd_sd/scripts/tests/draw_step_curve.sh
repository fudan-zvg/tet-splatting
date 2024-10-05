
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
RESUME='./logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt'
CFGLIST=(1. 2. 3. 4. 5. 6. 7. 8.)



for CFG in ${CFGLIST[@]}
    do
    python3 draw_ddim_cfg_curve.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_curve_1e-5.yaml  --scale_lr False --cfg $CFG --resume $RESUME --gpus 0,1,2,3,4,5,6,7,
done

# fid curves
python ./tools/compute_fid_pyiqa_curves.py --path ./logs/2023-05-29T06-00-43_sd_rgbd_1e-5/images/

# clip curves
python ./tools/compute_metric_pyiqa_curves.py --path ./logs/2023-05-29T06-00-43_sd_rgbd_1e-5/images/
