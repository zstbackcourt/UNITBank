#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
from init import *
import numpy as np

class GaussianNoiseLayer(nn.Module):
    def __init__(self, ):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        #print('gpu = ', x.data.get_device())
        gpu = x.data.get_device()
        noise = Variable(torch.randn(x.size()).cuda(gpu))
        return x + noise

class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride = 1):
        return nn.Conv2d(inplanes, out_planes, kernel_size = 3, stride = stride, padding = 1)

    def __init__(self, inplanes, planes, stride = 1, dropout = 0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += [self.conv3x3(inplanes, planes, stride)]
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace = True)]
        model += [self.conv3x3(planes, planes)]
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p = dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding = 0):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)]
        model += [nn.LeakyReLU(inplace = True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)

class LeakyReLUConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding = 0, output_padding = 0):
        super(LeakyReLUConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, bias = True)]
        model += [nn.LeakyReLU(inplace = True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)




