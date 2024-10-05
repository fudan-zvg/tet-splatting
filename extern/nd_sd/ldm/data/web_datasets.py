import multiprocessing
import os
import pdb
import sys
sys.path.append('./')
from functools import partial
from io import BytesIO

import albumentations
import cv2
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from libs.ControlNet_python_bings.annotator.normalbae import \
    NormalBaeBatchDetector
from libs.omnidata_torch.lib.midas import MidasBatchDetector
from libs.omnidata_torch.lib.utils import HWC3, resize_image
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def my_decoder(key, value):
    # solve the issue: https://github.com/webdataset/webdataset/issues/206

    if key.endswith('.jpg'):
        # return Image.open(BytesIO(value))
        return np.asarray(Image.open(BytesIO(value)).convert('RGB'))

    return None


class filter_fake:

    def __init__(self, punsafety=0.2, aest=4.5):
        self.punsafety = punsafety
        self.aest = aest

    def __call__(self, src):
        for sample in src:
            img, prompt, json = sample
            # watermark filter
            if json['pwatermark'] is not None:
                if json['pwatermark'] > 0.3:
                    continue

            # watermark
            if json['punsafe'] is not None:
                if json['punsafe'] > self.punsafety:
                    continue

            # watermark
            if json['AESTHETIC_SCORE'] is not None:
                if json['AESTHETIC_SCORE'] < self.aest:
                    continue

            # ratio filter
            w, h = json['width'], json['height']
            if max(w / h, h / w) > 3:
                continue

            yield img, prompt, json['AESTHETIC_SCORE']


'''
def filter_fake(src, punsafety =0.2, aest = 4.5):
    for sample in src:
        img, prompt, json = sample
        # watermark filter
        if json['pwatermark'] is not None:
            if json['pwatermark'] >0.3:
                continue

        # watermark
        if json['punsafe'] is not None:
            if json['punsafe'] >punsafety:
                continue
        # watermark
        if json['AESTHETIC_SCORE'] is not None:
            if json['AESTHETIC_SCORE'] <aest:
                continue

        # ratio filter
        w,h = json['width'], json['height']
        if max(w/h, h/w) > 3:
            continue

        yield img, prompt, json['AESTHETIC_SCORE']
'''


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Laion2b_Process(object):

    def __init__(self,
                 size=None,
                 degradation=None,
                 downscale_f=4,
                 min_crop_f=0.8,
                 max_crop_f=1.,
                 random_crop=True,
                 debug: bool = False):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        # downsacle_f = 0

        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False  # gets reset later if incase interp_op is from pillow

        if degradation == 'bsrgan':
            self.degradation_process = partial(
                degradation_fn_bsr, sf=downscale_f)

        elif degradation == 'bsrgan_light':
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
                'cv_nearest': cv2.INTER_NEAREST,
                'cv_bilinear': cv2.INTER_LINEAR,
                'cv_bicubic': cv2.INTER_CUBIC,
                'cv_area': cv2.INTER_AREA,
                'cv_lanczos': cv2.INTER_LANCZOS4,
                'pil_nearest': PIL.Image.NEAREST,
                'pil_bilinear': PIL.Image.BILINEAR,
                'pil_bicubic': PIL.Image.BICUBIC,
                'pil_box': PIL.Image.BOX,
                'pil_hamming': PIL.Image.HAMMING,
                'pil_lanczos': PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith('pil_')

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize,
                    size=self.LR_size,
                    interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)

    def __call__(self, samples):
        example = {}
        image, caption, aesthetics = samples

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(
            self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(
                height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(
                height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)['image']
        image = self.image_rescaler(image=image)['image']

        # -1, 1
        example['image'] = (image / 127.5 - 1.0).astype(np.float32)

        # depth prior is set to 384
        example['prior'] = resize_image(HWC3(image), 384)
        example['caption'] = caption

        return example


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False,
                 aest=4.5,
                 punsafety=0.1):

        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else min(
            batch_size * 2, multiprocessing.cpu_count())

        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            train_params = train['params']
            self.iter = int(20825 * 10000) // self.num_workers
            self.train_dataset = (
                wds.WebDataset(train_params['curls'], resampled=True).decode(
                    my_decoder, 'rgb8').shuffle(1000).to_tuple(
                        'jpg', 'txt', 'json').compose(
                            filter_fake(aest=aest, punsafety=punsafety)).map(
                                Laion2b_Process(
                                    size=train_params['size'],
                                    degradation='cv_bilinear',
                                    min_crop_f=train_params['min_crop_f'])
                            )  # we focus on high resoltion
                .with_epoch(self.iter))
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs['predict'] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def setup(self, stage=None):
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

        # for i in range(100):
        #     self.datasets['train'][-i]
        #     pdb.set_trace()

    def _train_dataloader(self):

        batch_size = self.batch_size
        num_workers = max(self.num_workers, self.batch_size)

        return wds.WebLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers).with_length(
                self.iter * self.num_workers / self.batch_size)

    def _val_dataloader(self, shuffle=False):

        if isinstance(self.datasets['validation'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn)


if __name__ == '__main__':

    midas_model = MidasBatchDetector().cuda()
    normal_model = NormalBaeBatchDetector().cuda()
    #20824 laion2b_5_2 = wds.WebDataset('../improved_aesthetics_5plus/laion-2ben-5_1/{00000..20824}.tar')
    urls = '../improved_aesthetics_5plus/laion-2ben-5_1/{00000..20824}.tar'

    dataset = (
        wds.WebDataset(urls, resampled=True).decode(
            my_decoder,
            'rgb8').shuffle(1000).to_tuple('jpg', 'txt', 'json').map(
                Laion2b_Process(
                    size=256, degradation='cv_bilinear',
                    min_crop_f=0.8))  # we focus on high resoltion
        .with_epoch(100000000000))

    loader = wds.WebLoader(
        dataset, batch_size=20, num_workers=20).with_length(40)

    cnt = 0

    for data in loader:

        print(data['image'].shape)

        prior_img = data['prior'].float().cuda()
        results = {}
        depth_pred = midas_model(prior_img)
        normal_pred = normal_model(prior_img)

        results.update(depth_pred)
        results.update(normal_pred)

        batch_size = prior_img.shape[0]

        for i in range(batch_size):
            prior = data['prior'].numpy()[i]
            normal = (results['normal'][i]).permute(1, 2, 0)
            depth = (results['depth'][i]).permute(1, 2, 0)
            i = batch_size * cnt + i

            prior_path = os.path.join('./prior_results',
                                      'img_{:04d}.png'.format(i))
            normal_path = os.path.join('./prior_results',
                                       'normal_{:04d}.png'.format(i))
            depth_path = os.path.join('./prior_results',
                                      'depth_{:04d}.png'.format(i))

            normal = ((normal + 1) / 2 * 255).detach().cpu().numpy().astype(
                np.uint8)
            depth = ((depth + 1) / 2 * 255).detach().cpu().numpy().astype(
                np.uint8)[..., 0]
            cv2.imwrite(prior_path, prior[..., ::-1])
            cv2.imwrite(normal_path, normal[..., ::-1])
            cv2.imwrite(depth_path, depth)
        cnt += 1
