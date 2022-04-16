from pytorch_lightning import LightningModule
from torch import nn
import torch


class conv_block(LightningModule):
    def __init__(self, in_channels, pool):
        super(conv_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3)
        self.conv_3 = nn.Conv2d(in_channels*4, in_channels*8, kernel_size=3)

        self.bn_1 = nn.BatchNorm2d(in_channels*2)
        self.bn_2 = nn.BatchNorm2d(in_channels*4)
        self.bn_3 = nn.BatchNorm2d(in_channels*8)

        if pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3)
        else:
            self.pool = nn.MaxPool2d(kernel_size=3)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activation(x)

        return x

class Regressor(LightningModule):
    def __init__(self, image_channels, num_features):
        super(Regressor, self).__init__()
    

        self.conv1 = nn.Sequential(
            conv_block(image_channels, 'avg'),
        )

        self.conv2 = nn.Sequential(
            conv_block(image_channels, 'max'),
        )

        self.gender = nn.Sequential(
            nn.Linear(2, 2400),
            nn.ReLU(),
            nn.Linear(2400, 6000),
            nn.ReLU()

        )

        self.graph_inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 2)]
        dim = torch.flatten(self.conv1(self.graph_inputs[0]), start_dim=1).size()[1]*2 + torch.flatten(self.gender(self.graph_inputs[1]), start_dim=1).size()[1]

        self.example_input_array = self.graph_inputs[0]


        self.regressor = nn.Sequential(
            nn.Linear(dim, 100),
        )

        self.out =  nn.Tanh()


    def forward(self, x, y):
        y = self.gender(y)
        y = torch.flatten(y, start_dim=1)

        x_1 = torch.clone(x)
        x_1 = self.conv1(x_1)
        x_1 = torch.flatten(x_1, start_dim=1)

        x_2 = torch.clone(x)
        x_2 = self.conv2(x_2)
        x_2 = torch.flatten(x_2, start_dim=1)

        x_3 = torch.cat((x_1, x_2), dim=1)

        z = torch.cat((x_3, y), dim=1)
        z = self.regressor(z)
  


        return z
