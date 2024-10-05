
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
RESUME_PATH=$1
RESUME=$1/checkpoints/last.ckpt
OUTPUT=$1/images
config='./configs/stable-diffusion/rgbd/sd_rgb_1_5_no_ema.yaml'
# ckpt_list=(4 6 7)
# sets=`ls $RESUME_PATH | grep -i step_ | sed "s:^:$RESUME_PATH/:`
ckpt_list=`ls $RESUME_PATH/checkpoints | grep -i step_0`
test_size=`ls $OUTPUT/ | grep -i test_0 | wc -l`
echo $RESUME
echo $OUTPUT
echo $ckpt_list
echo ${test_size}

declare -i cnt=0

for ckpt in ${ckpt_list[@]}
    do
    if [ "$cnt" -lt "$test_size"  ]; then
        echo "already predict"
    else
        echo $ckpt
        python3 draw_ddim_step_curve.py --base  $config --scale_lr False --resume_epoch $ckpt --resume $RESUME --gpus 0,1,2,3,4,5,6,7,
    fi
    cnt+=1
done

python3 ./tools/compute_fid_pyiqa_curves.py --path $OUTPUT  --total_size 30000 --type step
