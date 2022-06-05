from tkinter import S
from pytorch_lightning import LightningModule

from torch import nn
import torch

from utils.regressor import Regressor
from utils.extractor import ResNet50 as ResNet

class model(LightningModule):
    def __init__(self, img_channels, in_features, out_features):
        super(model, self).__init__()
        
        self.extractor = ResNet(img_channel = img_channels, num_classes = in_features)
        self.regressor = Regressor(input_size = in_features, output_size = out_features)

    def forward(self, x):
        x = self.extractor(x)
        x = self.regressor(x)
        
        return x
