import argparse
import glob
import os
import pdb

import cv2
import numpy as np


def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', '-i', type=str)
    parser.add_argument('--output_path', '-o', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parse()
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    rgb_path = sorted(glob.glob(os.path.join(input_path, '*_rgb.png')))
    depth_path = sorted(glob.glob(os.path.join(input_path, '*_depth.png')))

    for rgb_name, depth_name in zip(rgb_path, depth_path):
        rgb = cv2.imread(rgb_name)
        depth = cv2.imread(depth_name)
        cat_name = rgb_name.split('/')[-1].replace('_rgb', '')

        cat_img = np.concatenate([rgb, depth], axis=1)
        cat_path = os.path.join(output_path, cat_name)

        cv2.imwrite(cat_path, cat_img)
