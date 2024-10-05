"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import glob
import os
import os.path as osp
import pdb
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
sys.path.append('./')

import clip
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from ldm.data.laion_rgb_depth import COCO2KSRTest
from PIL import Image
from torch import autocast
from torch.utils.data import DataLoader, Dataset


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--batch-size', type=int, default=50, help='Batch size to use')
parser.add_argument(
    '--clip-model', type=str, default='ViT-L/14', help='CLIP model to use')
parser.add_argument(
    '--num-workers',
    type=int,
    help=('Number of processes to use for data loading. '
          'Defaults to `min(8, num_cpus)`'))
parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument(
    '--ref_path', type=str, default='', help='ref path to compute clip score')
parser.add_argument('--use_tag', action='store_true')

IMAGE_EXTENSIONS = {
    'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'
}

TEXT_EXTENSIONS = {'txt'}

args = parser.parse_args()


class cocorgbdataset(COCO2KSRTest):

    def __init__(self, **kwargs):
        ref_path = kwargs.pop('ref_path')
        self.preprocess = kwargs.pop('preprocess')
        self.ref_path = ref_path

        super().__init__(**kwargs)

    def __getitem__(self, i):
        example = self.sample(i)
        ref_path = os.path.join(self.ref_path, 'rgb_' + example['file_path_'])
        img_path = example['file_path_']
        img_path = example['file_path_'].replace('.npz', '.jpg')
        target_path = os.path.join(self.data_root, img_path)
        image = Image.open(target_path)
        ref_image = Image.open(ref_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if not ref_image.mode == 'RGB':
            image = ref_image.convert('RGB')

        image = self.preprocess(image)
        ref_image = self.preprocess(ref_image)

        caption = example['caption']

        return {
            'img': image,
            'ref_img': ref_image,
            'caption': caption,
        }


class ObjaverseDataset():

    def __init__(self,
                 size=256,
                 root='mvs_objaverse/',
                 debug=False,
                 random_crop=False,
                 preprocess=None,
                 use_tag=False,
                 folder_key='campos_256'):
        self.root = root
        self.size = size
        self.preprocess = preprocess
        objaverse_json = os.path.join(self.root, 'text_captions_cap3d.json')
        objaverse_meta = os.path.join(self.root,
                                      'objaverse_meta_nametags.json')
        self.folder_key = folder_key

        with open(objaverse_meta, 'r') as reader:
            self.objaverse_meta = json.load(reader)

        with open(objaverse_json, 'r') as reader:
            self.caption_objaverse = json.load(reader)
        self.orthogonal_view = [1, 7, 13, 19]
        self.items = sorted(self.caption_objaverse.keys())
        self.preprocess = preprocess

        self.use_tag = use_tag

    def __len__(self):
        return len(self.caption_objaverse.keys())

    def views(self):
        return len(self.orthogonal_view)

    def tag_name(self, key):
        folder, hash_id = key.split('/')
        img_folder = os.path.join(self.root, folder, hash_id, 'glb')
        glb_name = glob.glob(os.path.join(img_folder, '*.glb'))[0]
        glb_name = glb_name.split('/')[-1].replace('.glb', '')
        names = self.objaverse_meta[glb_name]
        if len(names[1]) > 0:
            caption = names[0] + ', ' + ', '.join(names[1])
        else:
            caption = names[0]
        return caption

    def __getitem__(self, i):

        try:
            key = self.items[i]
            # depose
            folder, hash_id = key.split('/')
            img_folder = os.path.join(self.root, folder, hash_id,
                                      self.folder_key)

            img_container = []
            prompts = []
            for view in self.orthogonal_view:
                img_path = os.path.join(img_folder,
                                        '{:05d}/{:05d}.png'.format(view, view))
                image = Image.open(img_path)

                if not image.mode == 'RGB':
                    image = image.convert('RGB')
                #[c, h, w]
                img = self.preprocess(image)
                img_container.append(img)

            # [C, h*views, w]
            img = torch.cat(img_container, dim=1)
            caption = self.tag_name(key)
        except:
            # 1 is okay
            key = self.items[1]

            # depose
            folder, hash_id = key.split('/')
            img_folder = os.path.join(self.root, folder, hash_id,
                                      self.folder_key)

            img_container = []
            prompts = []
            for view in self.orthogonal_view:
                img_path = os.path.join(img_folder,
                                        '{:05d}/{:05d}.png'.format(view, view))
                image = Image.open(img_path)

                if not image.mode == 'RGB':
                    image = image.convert('RGB')
                #[c, h, w]
                img = self.preprocess(image)
                img_container.append(img)

            # [C, h*views, w]
            img = torch.cat(img_container, dim=1)
            caption = self.tag_name(key)

        if not self.use_tag:
            caption = self.caption_objaverse[key]

        return {
            'img': img,
            'caption': caption,
            'key': key,
        }


class DummyDataset(Dataset):

    FLAGS = ['img', 'txt']

    def __init__(self,
                 real_path,
                 fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'img',
                 transform=None,
                 tokenizer=None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_foler = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_foler[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample

    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError('Got unexpected modality: {}'.format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True

    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
@autocast('cuda')
def calculate_clip_score(dataloader, model, tokenize, view_size):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    results = {}
    for batch_data in dataloader:
        captions = batch_data['caption']
        imgs = batch_data['img']
        keys = batch_data['key']

        merge_imgs = []
        merge_captions = []
        for img, caption in zip(imgs, captions):
            img = img[None]
            _, c, h, w = img.shape
            split_size = h // view_size
            split_imgs = torch.split(img, split_size, dim=2)
            merge_captions.extend([caption] * view_size)
            merge_imgs.extend(list(split_imgs))

        real = [tokenize(caption) for caption in merge_captions]
        real = torch.cat(real, dim=0).cuda()
        real_features = forward_txt(model, real)

        fake = torch.cat(merge_imgs, dim=0).cuda()
        fake_features = forward_modality(model, fake)
        # normalize features
        real_features = real_features / real_features.norm(
            dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(
            dim=1, keepdim=True).to(torch.float32)

        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum(-1)
        scores = score.view(-1, view_size)
        for key, score in zip(keys, scores):
            score = score.detach().cpu().numpy().tolist()
            results[key] = score

    with open('objaverse_clip_score_tag.json', 'w') as writer:
        json.dump(results, writer)

    return 0.


def forward_modality(model, data):
    features = model.encode_image(data)

    return features


def forward_txt(model, data):
    features = model.encode_text(data)

    return features


def main1():

    model, __, preprocess = open_clip.create_model_and_transforms(
        'ViT-g-14', pretrained='laion2b_s34b_b88k')
    tokenize = open_clip.get_tokenizer('ViT-g-14')
    print('Loading CLIP model: {}'.format('ViT-g-14'))
    model.cuda()

    dataset = ObjaverseDataset(
        size=256,
        root='mvs_objaverse/0/',
        debug=False,
        preprocess=preprocess,
        use_tag=args.use_tag,
        folder_key='campos_512_v4')

    dataset[0]

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    # 8 * 4
    dataloader = DataLoader(
        dataset, 8, num_workers=num_workers, pin_memory=True)
    dataloader = tqdm(dataloader)

    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model, tokenize,
                                      dataset.views())


def main():
    preprocess = None
    dataset = ObjaverseDataset(
        size=256,
        root='mvs_objaverse/0/',
        debug=False,
        preprocess=preprocess,
        folder_key='campos_512_v4')
    dataset[0]


if __name__ == '__main__':
    main1()
