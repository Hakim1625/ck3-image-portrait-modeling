from numpy import outer
import torch
import torch.nn as nn
import pytorch_lightning as pl

def init(layer):
    torch.nn.init.xavier_uniform(layer)

class block(pl.LightningModule):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)


        self.identity_downsample = identity_downsample
        self.stride = stride

        self.Elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.15)

        init(self.conv1.weight)
        init(self.conv2.weight)
        init(self.conv3.weight)

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.Elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.Elu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        x = self.dropout(x)
        x = self.Elu(x)

        return x

class l_block(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(l_block, self).__init__()
        self.l1 = nn.Linear(in_channels, in_channels)
        self.l2 = nn.Linear(in_channels, in_channels)
        self.l3 = nn.Linear(in_channels, out_channels)

        self.Elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.15)

        init(self.l1.weight)
        init(self.l2.weight)
        init(self.l3.weight)

    def forward(self, x):
        x1 = self.l1(x)
        x1 = self.Elu(x)
        x2 = self.l2(x1)
        x2 = self.Elu(x2)
        x3 = self.l3(x1 + x2)
        x3 = self.dropout(x3)
        x3 = self.Elu(x3)

        return x3


import utils.model_irse as irse
import utils.extract_feature_v2 as feature
from torchvision import models


class ResNet(pl.LightningModule):
    def __init__(self, block, layers, image_channels, num_features):
        super(ResNet, self).__init__()


        #//// Extracting Facial features with the help of a pretrained network
        model_root = './utils/backbone_ir50_asia.pth'
        self.backbone = irse.IR_50((112, 112))
        self.backbone.load_state_dict(torch.load(model_root))

        self.feature_extractor = lambda image, backbone: feature.extract_feature(image, backbone)



        #////// Predicting Gender with the help of a pretrained network
        #self.gender_predictor = models.resnet18(pretrained=True)
        #num_features = self.gender_predictor.fc.in_features
        #self.gender_predictor.fc = nn.Linear(num_features, 2)
        #self.gender_predictor.load_state_dict(torch.load('./utils/face_gender_classification_transfer_learning_with_ResNet18(1).pth'))
        #self.gender_predictor.to('cuda')

        #self.softmax = nn.Softmax(dim=1)

        self.embedding = nn.Sequential(
            l_block(512, 654), 
            l_block(654, 886)
        )



        #////// Rest of the Resnet
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.Elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)






        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features=((512 * 4) + 886), out_features=100)
        self.out = nn.Tanh()
    

        init(self.conv1.weight)
        init(self.linear.weight)

    def forward(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.Elu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        e = self.feature_extractor(image, self.backbone)

        #g = self.gender_predictor(image)
        e = self.embedding(e)

        #c = e+g

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, e), dim=1)
        x = self.linear(x)
        x = self.out(x)
        x = x*265

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_features=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_features)


def ResNet101(img_channel=3, num_features=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_features)


def ResNet152(img_channel=3, num_features=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_features)


def main():
    model = ResNet50(img_channel=6, num_features=101)
    y = model(torch.randn(1, 6, 224, 224)).to('cuda')
    print(y)


if __name__ == "__main__":
    main()