import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

# Structure for ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, classCount, isTrained):
      super(ResNet50, self).__init__()
      self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)
      kernelCount = self.resnet50.fc.in_features
      self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x

# Structure for AlexNet model
class AlexNet(nn.Module):
    def __init__(self, classCount, isTrained):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=isTrained)
        kernelCount = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.alexnet(x)
        return x

# Structure for VGG16 model
class VGG16(nn.Module):
    def __init__(self, classCount, isTrained):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=isTrained)
        kernelCount = self.vgg16.classifier[6].out_features
        self.Linear = nn.Linear(kernelCount,classCount)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg16(x)
        x = self.Linear(x)
        x = self.Sigmoid(x)
        return x

# Structure for DenseNet121 model
class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

# Structure for DenseNet161 model
class DenseNet161(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet161, self).__init__()
        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)
        kernelCount = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet161(x)
        return x

# Structure for DenseNet169 model
class DenseNet169(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet169, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x

# Structure for DenseNet201 model
class DenseNet201(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet201, self).__init__()
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x