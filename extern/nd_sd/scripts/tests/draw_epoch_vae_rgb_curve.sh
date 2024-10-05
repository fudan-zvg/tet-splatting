
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
RESUME_PATH=$1
RESUME=$1/checkpoints/step_0000000.ckpt
OUTPUT=$1/images

config='./configs/autoencoder/sd_1_5_rgb.yaml'
# sets=`ls $RESUME_PATH | grep -i step_ | sed "s:^:$RESUME_PATH/:`
ckpt_list=`ls $RESUME_PATH/checkpoints | grep -i step_00`
echo $ckpt_list

for ckpt in ${ckpt_list[@]}
    do
    python3 draw_vae_epoch_curve.py --base  $config --scale_lr False --resume_epoch $ckpt --resume $RESUME --gpus 0,1,2,3,4,5,6,7,
done

python3 ./tools/compute_fid_pyiqa_curves.py --path $OUTPUT  --total_size 30000 --type step
