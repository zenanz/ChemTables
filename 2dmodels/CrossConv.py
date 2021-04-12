import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(CrossConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self._mask = torch.ones(self.weight.size())

        center_h = self.weight.size(2) // 2 # kernel_h // 2
        center_w = self.weight.size(3) // 2 # kernel_w // 2

        self._mask[:, :, :center_h-1, :center_w-1] = 0 # mask top left
        self._mask[:, :, center_h+1:, :center_w-1] = 0 # mask bottom left

        self._mask[:, :, :center_h-1, center_w+1:] = 0 # mask top right
        self._mask[:, :, center_h+1:, center_w+1:] = 0 # mask bottom right

        self._mask = nn.Parameter(self._mask, requires_grad=False)


    def forward(self, inputs):

        self.weight = nn.Parameter(self.weight * self._mask)

        return F.conv2d(inputs, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
