import argparse
import glob
import os
import pdb
import random
import time
from multiprocessing import Pool, Process


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument(
        '--model', type=str, default='zoe', choices=['midas', 'zoe'])

    args = parser.parse_args()
    return args


def apply_depth_predict(gpu_id, start, end):
    if args.model == 'zoe':
        command = 'CUDA_VISIBLE_DEVICES={} python3 ./tools/compute_zoe_metric.py --path {} --start {} --end {}'.format(
            gpu_id, args.path, start, end)
    elif args.model == 'midas':
        command = 'CUDA_VISIBLE_DEVICES={} python3 ./tools/compute_midas_metric.py --path {} --start {} --end {}'.format(
            gpu_id, args.path, start, end)
    print(command)
    os.system(command)


if __name__ == '__main__':

    args = get_parse()
    predict_depth = glob.glob(
        os.path.join(args.path, '{}*.npz'.format(args.model)))

    if len(predict_depth) == 6000:
        exit()

    pool = Pool(args.gpu)  #创建一个5个进程的进程池
    gpu_idx = 0
    num_gpu = args.gpu
    steps = 6000 // num_gpu

    ranges = [[i * steps, (i + 1) * steps] for i in range(num_gpu)]
    ranges[-1][-1] = 6000

    for part_i in range(num_gpu):
        start, end = ranges[part_i]
        pool.apply_async(func=apply_depth_predict, args=(part_i, start, end))

    pool.close()
    pool.join()
    print('end')
