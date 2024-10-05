
# python3 vae_main.py --base ./configs/stable-diffusion/rgbd/sd_rgbd_1e-5.yaml  --scale_lr False --resume './logs/2023-05-29T06-00-43_sd_rgbd_1e-5/checkpoints/last.ckpt' --gpus 0,1,2,3,4,5,6,7,
RESUME_PATH=$1
RESUME=$1/checkpoints/step_00001250.ckpt
OUTPUT=$1/images
config='./configs/stable-diffusion/rgbd/sd_1_5/rgbd_laion2b_inference.yaml'
# ckpt_list=(4 6 7)
# sets=`ls $RESUME_PATH | grep -i step_ | sed "s:^:$RESUME_PATH/:`
ckpt_list=`ls $RESUME_PATH/checkpoints | grep -i step_0`
test_size=`ls $OUTPUT/ | grep -i test_0 | wc -l`
echo $RESUME
echo $OUTPUT
# ckpt_list=(step_00087500.ckpt step_00100000.ckpt)
# ckpt_list=()
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

exit

# compute depth metric

echo "compute depth metric"
test_folders=`ls $OUTPUT | grep -i test_0 | sed "s:^:$OUTPUT/:"`


for test_folder in ${test_folders[@]}
do
    echo $test_folder
    python ./tools/compute_zoe_metric_distributed.py --path $test_folder --gpu 8
    # infer midas_depth
    python ./tools/compute_zoe_metric_distributed.py --path $test_folder --gpu 8 --model midas
done


for test_folder in ${test_folders[@]}
do
    python3 ./tools/compute_depth_metric.py --path $test_folder
done


## draw clip metric
