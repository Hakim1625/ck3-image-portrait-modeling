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

        self.activation = nn.ReLU()

        init(self.conv1.weight)
        init(self.conv2.weight)
        init(self.conv3.weight)

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        x = self.activation(x)

        return x

class l_block(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(l_block, self).__init__()
        self.l1 = nn.Linear(in_channels, in_channels)
        self.l2 = nn.Linear(in_channels, in_channels)
        self.l3 = nn.Linear(in_channels, out_channels)

        self.activation = nn.ReLU()

        init(self.l1.weight)
        init(self.l2.weight)
        init(self.l3.weight)

    def forward(self, x):
        x1 = self.l1(x)
        x1 = self.activation(x)
        x2 = self.l2(x1)
        x2 = self.activation(x2)
        x3 = self.l3(x1 + x2)
        x3 = self.activation(x3)

        return x3

class ResNet(pl.LightningModule):
    def __init__(self, block, layers, image_channels, num_features):
        super(ResNet, self).__init__()
        self.example_input_array=torch.randn(1, 3, 244, 244)
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

        self.gender = l_block(2, 6000)
        self.regressor = nn.Linear(in_features=(512 * 4)+6000, out_features=100)






        init(self.conv1.weight)
        init(self.regressor.weight)

    def forward(self, image, gender):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.Elu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        y = self.gender(gender)

        z = torch.cat((x, y), dim=1)
        z = self.regressor(z)

        return z

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

if __name__ == '__main__':
    None