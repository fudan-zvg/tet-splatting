import argparse
import glob
import os
import pdb
import sys
sys.path.append('./')

import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy import linalg
from tools.compute_fid_pyiqa import FID
from torch import nn
from tqdm import tqdm


warnings.filterwarnings('ignore')

marker = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h',
    'H', '+', 'x', 'D', 'd', '|', '_', '.', ','
]


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', default='', help='train or test', type=str)
    parser.add_argument(
        '--total_size', default=30000, help='train or test', type=int)
    parser.add_argument(
        '--type', default='epoch', help='train or test', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parse()
    data = pd.read_parquet(f'./coco/subset.parquet')
    base = data['file_name'].to_numpy()[:args.total_size]

    save_path = os.path.join('./outputs/fid_curve', args.path.split('/')[2])
    os.makedirs(save_path, exist_ok=True)

    metric_path = os.path.join(save_path, '{}_metric.npz'.format(args.type))
    if os.path.exists(metric_path):
        metric_matrix = np.load(metric_path)
        y_axis = metric_matrix['y_axis'].tolist()
    else:
        y_axis = []

    folder_chain = sorted(glob.glob(os.path.join(args.path, 'test_00*')))
    x_aixs = [
        float(folder.split('/')[-1].split('_')[-1]) for folder in folder_chain
    ]
    folder1 = [os.path.join('./coco/val2014/', name) for name in base]
    fid_metric = FID()
    fid_metric.cuda()

    for folder_name in tqdm(folder_chain[len(y_axis):]):
        folder2 = [os.path.join(folder_name, 'rgb_' + name) for name in base]

        try:
            scores = fid_metric(folder1, folder2)
            y_axis.append(float(scores))
        except:
            break

    x_aixs = x_aixs[:len(y_axis)]
    plt.plot(
        x_aixs,
        y_axis,
        color='blue',
        linewidth=2,
        marker='v',
        markersize=10,
        markerfacecolor='red',
        markeredgewidth=1,
        markeredgecolor='red')

    plt.grid(True)
    plt.ylabel('FID')
    if args.type == 'cfg':
        plt.xlabel('classifier-free guidance')
        plt.savefig(os.path.join(save_path, 'fid_cfg.png'))
    elif args.type == 'epoch':
        plt.xlabel('epoch')
        plt.savefig(os.path.join(save_path, 'fid_epcoh.png'))
    elif args.type == 'step':
        plt.xlabel('')
        plt.savefig(os.path.join(save_path, 'fid_step.png'))
    else:
        raise NotImplemented

    print(y_axis)
    np.savez(metric_path, x_aixs=x_aixs, y_axis=y_axis)
