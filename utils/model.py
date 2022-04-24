from utils.gene_dicts import get_lengths

from pytorch_lightning import LightningModule

from torch import batch_norm, nn
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

        self.activation = nn.ELU()

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

class res_block(LightningModule):
    def __init__(self, in_channels, out_channels):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(in_channels, in_channels)
        self.l2 = nn.Linear(in_channels, out_channels)
        self.l3 = nn.Linear(out_channels, out_channels)
        self.l4 = nn.Linear(out_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x_1 = self.l1(x)
        x_1 = self.bn1(x_1)
        x_1 = self.activation(x_1)

        x_1 = self.l2(x_1)
        x_1 = self.bn2(x_1)

        x_2 = self.l3(torch.clone(x_1))
        x_2 = self.activation(x_2)

        x_2 = self.l4(x_2)
        x_2 = self.sigmoid(x_2)

        x_3 = x_2 * x_1
        x_3 = self.activation(x_3)

        return x_3

class Regressor(LightningModule):
    def __init__(self, embeddings_size=2622, lengths=get_lengths()):
        super(Regressor, self).__init__()
        
        self.regressor = nn.Sequential(
            res_block(embeddings_size, 1000),
            res_block(1000, 1000),
            res_block(1000, 1000),
            res_block(1000, 1000),
            nn.Linear(1000, 220),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.regressor(x)
        x =  x*265
        
        return x


if __name__ == '__main__':
    model = Regressor()
    
