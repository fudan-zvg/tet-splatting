import glob
import os
import pdb
import sys
sys.path.append('./')

import cv2
import numpy as np
from tqdm import tqdm



def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append(
            '%f %f %f %d %d %d 0\n' %
            (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, 'w')
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), ''.join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale

    X = (u - K[0, 2])
    Y = (v - K[1, 2])

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    Z = Z * 512
    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points


# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(scale, view_ply_in_world_coordinate):

    imgs = glob.glob('./outputs/txt2img-samples/rgbd/*.png')

    if view_ply_in_world_coordinate:
        poses = np.fromfile(
            os.path.join(dataset_path, 'poses.txt'), dtype=float, sep='\n ')
        poses = np.reshape(poses, newshape=(-1, 4, 4))
    else:
        poses = np.eye(4)

    for i in tqdm(range(0, len(imgs))):

        img = cv2.imread(imgs[i])
        h, w, _ = img.shape
        K = np.eye(3)
        K[0, 0] = 255
        K[1, 1] = 255
        K[0, 2] = 256
        K[1, 2] = 256

        rgb = img[:, :w // 2, :]
        depth = img[:, w // 2:]
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        depth = 255 - depth
        depth = depth / 255

        # depth = (depth -depth.min()) / (depth.max() -depth.min())
        # depth = 1. / depth

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(
                rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(
                rgb, depth, scale=scale, K=K, pose=poses)

        save_ply_name = imgs[i].split('/')[-1].replace('.png', '.ply')
        save_ply_path = os.path.join('outputs/depth_view/', 'point_clouds')

        if not os.path.exists(save_ply_path):
            os.makedirs(save_ply_path, exist_ok=True)
        write_point_cloud(
            os.path.join(save_ply_path, save_ply_name), current_points_3D)


if __name__ == '__main__':
    view_ply_in_world_coordinate = False
    scale_factor = 1.0
    build_point_cloud(scale_factor, view_ply_in_world_coordinate)
