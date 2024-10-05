# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation
# https://github.com/baegwangbin/surface_normal_uncertainty

import os
import pdb
import types

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange

from .models.NNET import NNET
from .utils import utils


class NormalBaeDetector:

    def __init__(self):
        modelpath = os.path.join(
            './libs/ControlNet-v1-1-nightly/annotator/normalbae/',
            'scannet.pt')
        assert os.path.exists(modelpath)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = utils.load_checkpoint(modelpath, model)
        model = model.cuda()
        model.eval()
        self.model = model
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def eval(self):
        self.model.eval()

    def __call__(self, input_image):

        with torch.no_grad():
            image_normal = input_image / 255.0
            image_normal = rearrange(image_normal, 'b h w c -> b c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)[0][-1][:, :3, ...]

            return {'normal': normal.clip(min=-1., max=1.)}


class NormalBaeBatchDetector(nn.Module):

    def __init__(self):
        super().__init__()

        modelpath = os.path.join(
            './libs/ControlNet-v1-1-nightly/annotator/normalbae/',
            'scannet.pt')
        assert os.path.exists(modelpath)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = utils.load_checkpoint(modelpath, model)
        model = model.cuda()
        model.eval()
        self.model = model
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input_image):

        with torch.no_grad():
            image_normal = input_image / 255.0
            image_normal = rearrange(image_normal, 'b h w c -> b c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)[0][-1][:, :3, ...]

            return {'normal': normal.clip(min=-1., max=1.)}
