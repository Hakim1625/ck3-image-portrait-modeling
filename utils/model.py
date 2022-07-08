from pytorch_lightning import LightningModule

from utils.regressor import Regressor
from utils.extractor import ResNet50 as ResNet
from torchvision import models
import torch

class mask_rcnn(LightningModule):
    def __init__(self):
        super(mask_rcnn, self).__init__()
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to('cuda').eval()

    def forward(self, batch):
        with torch.no_grad():
            masks = self.model(batch)
            masks_batch = [mask.get('masks')[0][0] for mask in masks]
            return torch.stack(masks_batch)


class model(LightningModule):
    def __init__(self, img_channels, in_features, out_features):
        super(model, self).__init__()

        self.mask = mask_rcnn()
        self.extractor = ResNet(img_channel = img_channels+1, num_classes = in_features)
        self.regressor = Regressor(input_size = in_features, output_size = out_features)

    def forward(self, x):
        y = self.mask(x.clone(x))
        x = torch.stack((x, y), dim=1)

        x = self.extractor(x)
        x = self.regressor(x)
        
        return x
