from pytorch_lightning import LightningModule
from torch import nn
import torch


class conv_block(LightningModule):
    def __init__(self, in_channels):
        super(conv_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=6, stride=2)
        self.conv_2 = nn.Conv2d(in_channels*2, in_channels*4, kernel_size=6, stride=2)
        self.conv_3 = nn.Conv2d(in_channels*4, in_channels*8, kernel_size=6, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))

        return x



class Regressor(LightningModule):
    def __init__(self, image_channels, num_features):
        super(Regressor, self).__init__()
        self.graph_inputs = [torch.randn(1, 3, 224, 224), torch.randn(1, 2)]

        self.relu = nn.ReLU()

        self.conv = nn.Sequential(
            conv_block(image_channels),
        )

        self.gender = nn.Sequential(
            nn.Linear(2, 1200),
            nn.ReLU(),
            nn.Linear(1200, 6000)
        )

        dim = torch.flatten(self.conv(self.graph_inputs[0]), start_dim=1).size()[1]
        + torch.flatten(self.gender(self.graph_inputs[1]), start_dim=1).size()[1]


        self.regressor = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),

        )


    def forward(self, x, y):
        y = self.gender(y)
        y = torch.flatten(y, start_dim=1)

        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)

        z = torch.cat((x, y), dim=1)
        z = self.regressor(z)
        return z
