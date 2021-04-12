import torch
import torchvision.models as models
from CrossConv import CrossConv2d

class TBResNet(models.ResNet):

    def __init__(self,
                block,
                layers,
                zero_init_residual=False,
                groups=1,
                width_per_group=64,
                replace_stride_with_dilation=None,
                norm_layer=None,
                inplane=None,
                crossconv=None,
                dropout=0):

        super(TBResNet, self).__init__(block, layers, zero_init_residual=False,
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None)

        del self.fc # discard fully connected layer in resnet
        self._output_dim = 512 * block.expansion
        self.inputconv = torch.nn.Conv2d(inplane,
                                inplane, kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        if crossconv is not None:
            self._h_limit, self._w_limit = crossconv
            # self._input_conv = CrossConv2d(inplane,
            #                             inplane,
            #                             kernel_size = (self._h_limit * 2 - 1, self._w_limit * 2 - 1),
            #                             stride=1,
            #                             padding=(self._h_limit - 1 , self._w_limit - 1))

        # self.bn0 = torch.nn.LayerNorm([inplane, self._h_limit, self._w_limit])
        self.bn0 = torch.nn.BatchNorm2d(inplane)

        # Modify first convolution layer to fit embedder output size
        self.conv1 = torch.nn.Conv2d(inplane,
                                64, kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False)

        self.dp = torch.nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.inputconv(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dp(x)


        return x


def TBResNet18(inplane, crossconv=None, dropout=0):
    return TBResNet(models.resnet.BasicBlock, [2, 2, 2, 2], inplane=inplane, dropout=dropout, crossconv=crossconv)
