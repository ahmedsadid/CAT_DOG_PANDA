import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset, random_split
import PIL
from PIL import Image

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def cn_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    pass

class ResNet(ImageClassificationBase):
  def __init__(self, in_channels, out_classes):
    super().__init__()

    self.conv1 = cn_block(in_channels, 64)
    self.conv2 = cn_block(64, 128, pool=True)
    self.res1 = nn.Sequential(cn_block(128,128),
                              cn_block(128,128))
    
    self.conv3 = cn_block(128, 256, pool=True)
    self.conv4 = cn_block(256, 512, pool=True)
    self.res2 = nn.Sequential(cn_block(512,512),
                              cn_block(512,512))
    self.conv5 = cn_block(512, 256, pool=True)
    self.res3 = nn.Sequential(cn_block(256,256),
                              cn_block(256,256))
    
    self.classifier = nn.Sequential(nn.MaxPool2d(2), 
                                    nn.Flatten(), 
                                    nn.Dropout(0.2),
                                    nn.Linear(256*3*3, out_classes))
    
  def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


resnetModel = ResNet(3,3)
