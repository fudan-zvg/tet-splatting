import argparse
import glob
import os
import os.path as osp
import pdb
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('./')


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str)

    args = parser.parse_args()
    return args


class Metric(object):

    def __init__(self):
        self.count = 0
        self.val = 0.

    def update(self, v, n):
        self.val += v
        self.count += n

    @property
    def values(self):
        return (self.val / self.count).item() if self.count > 0 else 0.


class DepthMetric(object):

    def __init__(self, pred_files, midas_files, ref_files, save_path):
        self.ref_files = ref_files
        self.depth_files = pred_files
        self.midas_files = midas_files
        self.save_path = save_path

    def compute_values(self, pred_disparity, ref_depth):
        valid_mask = ref_depth > 0.
        ref_disparity = 1. / ref_depth
        scale, shift = self.compute_scale_and_shift(
            pred_disparity.squeeze(1), ref_disparity.squeeze(1),
            valid_mask.squeeze(1))
        scale_pred_disparity = scale.view(
            -1, 1, 1) * pred_disparity + shift.view(-1, 1, 1)
        scale_pred_depth = torch.clamp(
            1. / (scale_pred_disparity + 1e-9), min=0, max=ref_depth.max())

        abs_error = (
            (scale_pred_depth[valid_mask] - ref_depth[valid_mask]).abs() /
            (ref_depth[valid_mask]) + 1e-6).mean()
        rmse_error = torch.sqrt((((scale_pred_depth[valid_mask]
                                   - ref_depth[valid_mask]))**2).mean())

        return abs_error, rmse_error

    def compute_metrics(self, key='disparity'):

        avg = 0.
        abs_ref = Metric()
        rmse_ref = Metric()
        midas_abs_ref = Metric()
        midas_rmse_ref = Metric()

        pbar = tqdm(
            zip(self.depth_files, self.midas_files, self.ref_files),
            desc=f'abs_ref:{abs_ref.values}, rmse_ref:{rmse_ref.values},' +
            f'midas_abs_ref:{midas_abs_ref.values}, midas_rmse_ref:{midas_rmse_ref.values}\n'
        )
        for pred, midas, ref in pbar:
            ref_depth = np.load(ref)['values'][0, 0]
            midas_depth = np.load(midas)['values']

            pred_disparity = cv2.imread(pred)[..., 0]
            pred_disparity = pred_disparity.astype(np.float32)
            h, w = pred_disparity.shape

            ref_depth = torch.from_numpy(ref_depth).float()[None, None]
            midas_disparity = torch.from_numpy(midas_depth).float()[None, None]

            pred_disparity = torch.from_numpy(pred_disparity).float()[
                None, None].cuda()
            ref_depth = F.interpolate(
                ref_depth, size=(w, h), mode='bilinear',
                align_corners=True).cuda()
            midas_disparity = F.interpolate(
                midas_disparity,
                size=(w, h),
                mode='bilinear',
                align_corners=True).cuda()

            abs_error, rmse_error = self.compute_values(
                pred_disparity, ref_depth)

            abs_ref.update(abs_error, 1)
            rmse_ref.update(rmse_error, 1)

            abs_error, rmse_error = self.compute_values(
                midas_disparity, ref_depth)
            midas_abs_ref.update(abs_error, 1)
            midas_rmse_ref.update(rmse_error, 1)

            pbar.set_description(
                desc=f'abs_ref:{abs_ref.values}, rmse_ref:{rmse_ref.values},' +
                f'midas_abs_ref:{midas_abs_ref.values}, midas_rmse_ref:{midas_rmse_ref.values}\n'
            )

        return abs_ref.values, rmse_ref.values, midas_abs_ref.values, midas_rmse_ref.values

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (
            det[valid] + 1e-6)
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (
            det[valid] + 1e-6)

        return x_0, x_1


if __name__ == '__main__':
    args = get_parse()
    save_path = os.path.join('./outputs/fid_curve', args.path.split('/')[2])
    os.makedirs(save_path, exist_ok=True)

    metric_path = os.path.join(save_path, 'depth_metric.csv')
    epoch = int(args.path.split('/')[-1].split('_')[-1])

    if os.path.exists(metric_path):
        data = pd.read_csv(metric_path).values.tolist()
        if data[-1][0] >= epoch:
            exit()
    else:
        data = []

    depth_files = sorted(glob.glob(osp.join(args.path, '*.npz')))
    zoe_files = list(filter(lambda x: 'zoe_COCO' in x, depth_files))
    midas_files = list(filter(lambda x: 'midas_COCO' in x, depth_files))

    pred_files = [
        zoe_file.replace('zoe_COCO', 'depth_COCO').replace('.npz', '.jpg')
        for zoe_file in zoe_files
    ]

    metric_computer = DepthMetric(pred_files, midas_files, zoe_files,
                                  args.path)
    abs_ref, rmse_ref, midas_abs_ref, midas_rmse_ref = metric_computer.compute_metrics(
    )
    new_colume = [epoch, abs_ref, rmse_ref, midas_abs_ref, midas_rmse_ref]
    data.append(new_colume)
    df1 = pd.DataFrame(
        data=data,
        columns=[
            'epoch', 'abs_ref', 'rmse_ref', 'midas_abs_ref', 'midas_rmse_ref'
        ])
    df1.to_csv(metric_path, index=False)
