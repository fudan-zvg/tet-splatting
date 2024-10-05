import os
import pdb
import sys

import cv2
import numpy as np
import numpy as np
from tqdm import tqdm

sys.path.append('./')


def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append('%f %f %f %d %d %d 0\n' %
                          (X, Y, Z, color[0], color[1], color[2]))

    print(ply_file)
    file = open(ply_file, 'w')
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), ''.join(points)))
    file.close()
    print('save ply, fx:{}, fy:{}, cx:{}, cy:{}'.format(fx, fy, cx, cy))


if __name__ == '__main__':

    img = cv2.imread('./outputs/txt2img-samples/rgbd/grid-0000.png',
                     cv2.IMREAD_UNCHANGED)
    save_path = './outputs/depth_view/debug.ply'

    h, w, c = img.shape
    rgb = img[:, :512, :]
    disparity = img[:, 512:, 0]
    # disparity = cv2.cvtColor( disparity,cv2.COLOR_BGR2GRAY)
    # depth = 256-disprity

    cv2.imwrite('fuck.png', rgb)
    cv2.imwrite('fuck_1.png', disparity)
    cv2.imwrite('fuck_2.png', img)

    pdb.set_trace()

    fx = fy = depth.shape[1] * 1.2
    cx = depth.shape[1] / 2
    cy = depth.shape[0] / 2
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    generate_pointcloud(rgb, depth, save_path, intr=intr, scale=1.0)
