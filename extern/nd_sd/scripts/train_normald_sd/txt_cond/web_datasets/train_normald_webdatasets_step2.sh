unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

python3 main.py --base ./configs/stable-diffusion/normald/sd_1_5/txt_cond/web_datasets/laion_2b_step2.yaml -t --scale_lr False \
  data.params.train.params.curls='../improved_aesthetics_5plus/laion-2ben-5_0/{00000..60580}.tar' ${@:1}
