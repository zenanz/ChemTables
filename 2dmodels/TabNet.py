import torch
import torch.nn.functional as F
from CrossConv import CrossConv2d

class ResidualBlock(torch.nn.Module):
    def __init__(self, F, kernel_size=(3,3), padding=(1,1)):
        super(ResidualBlock, self).__init__()
        self.conv = torch.nn.Conv2d(F, F, kernel_size=(3, 3), stride=1, padding=(1,1), bias=False)
        # self.conv = CrossConv2d(F, F, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(F)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        F_x = self.conv(inputs)
        F_x = self.bn(F_x)
        F_x = self.relu(F_x + inputs)
        return F_x

class TabNet(torch.nn.Module):
    def __init__(self, inplane, table_size_limit):
        super(TabNet, self).__init__()
        # Add table embedder before feeding input to ResNet
        # Modify first convolution layer to fit embedder output size
        F_dim = 32
        h_limit, w_limit = table_size_limit
        self.conv0 = torch.nn.Conv2d(inplane, F_dim, kernel_size=(3, 3), stride=1, padding=(1,1), bias=False)
        self.bn0 = torch.nn.BatchNorm2d(F_dim)

        self.layer1 = ResidualBlock(F_dim)
        self.layer2 = ResidualBlock(F_dim)
        self.layer3 = ResidualBlock(F_dim)

        self._output_dim = h_limit*w_limit*F_dim

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)

        return x
