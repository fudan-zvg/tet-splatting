# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
image_folder='./logs/2023-06-08T14-02-34_autoencoder_kl_rgbd_1e-5_high_resolu'
RESUME=$image_folder/checkpoints/last.ckpt
# RESUME='./logs/2023-06-08T13-56-47_autoencoder_kl_rgbd_1e-5_laion2b/checkpoints/last.ckpt'
config='./configs/autoencoder_rgbd/draw_laion2b_vae_epoch_steps.yaml'
# ckpt_list=(4 6 7)
RESUME_PATH=$1
# sets=`ls $RESUME_PATH | grep -i step_ | sed "s:^:$RESUME_PATH/:`
ckpt_list=`ls $RESUME_PATH | grep -i step_`
echo $ckpt_list


ckpt_list=(step_00050000.ckpt step_00062500.ckpt)

for ckpt in ${ckpt_list[@]}
    do
    # python3 draw_vae_epoch_curve.py --base  ./configs/autoencoder_rgbd/draw_vae_epoch_curve.yaml --scale_lr False --resume_epoch $ckpt --resume $RESUME --gpus 0,1,2,3
    python3 draw_vae_epoch_curve.py --base  $config --scale_lr False --resume_epoch $ckpt --resume $RESUME --gpus 0,1,2,3,
done

echo $image_folder/images
python3 ./tools/compute_fid_pyiqa_curves.py --path $image_folder/images --total_size 30000 --type step
