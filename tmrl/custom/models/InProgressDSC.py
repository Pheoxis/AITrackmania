import torch
from collections import OrderedDict

from cv2 import imshow
from torch import optim, nn

import numpy as np
from torchvision import datasets, transforms
import helper

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self,hidden_units: list,output_units: int,use_gpu : bool = False):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1, groups=hidden_units)
        self.pointwise = nn.Conv2d(hidden_units, output_units, kernel_size=1)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model=createmodel()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = nn.ReLU()(x)
        return x

class DepthwiseSeparableConvNet(nn.Module):
    def init(self, hidden_units: list, output_units: int,num_classes=10, image_size=28, num_blocks=3):
        super(DepthwiseSeparableConvNet, self).init()
        self.hidden_units = hidden_units
        self.output_units= output_units
        self.features = []
        for _ in range(num_blocks):
            self.features.append(DepthwiseSeparableConvBlock(in_channels, out_channels))
            self.features.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = 2
        self.features = nn.Sequential(self.features)



class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DeepwiseConvolutionX6(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=6):
        super(DeepwiseConvolutionX6, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(DepthwiseSeparableConv2d(in_channels, out_channels))
            else:
                layers.append(DepthwiseSeparableConv2d(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.avg_pool(x)
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x