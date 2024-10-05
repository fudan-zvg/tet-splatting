import glob
import os

midas_folders = sorted(glob.glob('./laion_art_depth/midas/part*'))
midas_writer = './laion_art_depth/midas_tokens.txt'

for cnt, midas_folder in enumerate(midas_folders):
    part_name = midas_folder.split('/')[-1]

    if cnt == 0:
        cmd = 'ls {} |grep -i .npz | sed \"s:^:{}/:\" > {}'.format(
            midas_folder, part_name, midas_writer)
    else:
        cmd = 'ls {} |grep -i .npz | sed \"s:^:{}/:\" >> {}'.format(
            midas_folder, part_name, midas_writer)

    os.system(cmd)
    print(cmd)
