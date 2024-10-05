# python ./tools/compute_zoe_metric_distributed.py --path ./logs/2023-05-29T06-00-43_sd_rgbd_1e-5/images/test --gpu 4

# infer zoe_depth
python ./tools/compute_zoe_metric_distributed.py --path $1 --gpu $2
# infer midas_depth
python ./tools/compute_zoe_metric_distributed.py --path $1 --gpu $2 --model midas

# compute vie
python ./tools/compute_depth_metric.py --path $1
