import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self._conv_2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self._conv_1(x)
        x = self._conv_2(x)
        return x


class UNet(nn.Module):
    def __init__(self, out_channels, num_features):
        super().__init__()
        self._conv_down_1 = ConvBlock(3, out_channels)
        self._conv_down_2 = ConvBlock(out_channels, out_channels * 2)
        self._conv_down_3 = ConvBlock(out_channels * 2, out_channels * 4)
        self._conv_bottom = ConvBlock(out_channels * 4, out_channels * 8)

        self._conv_up_1 = ConvBlock(out_channels * 12, out_channels * 4)
        self._conv_up_2 = ConvBlock(out_channels * 6, out_channels * 2)
        self._conv_up_3 = ConvBlock(out_channels * 3, out_channels)

        self._conv_heatmap = nn.Sequential(nn.Conv2d(out_channels, num_features, 3, padding='same'),
                                           nn.Sigmoid())

        self._max_pool = nn.MaxPool2d(2)
        self._up_sample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        conv_down_1 = self._conv_down_1(x)
        conv_down_2 = self._conv_down_2(self._max_pool(conv_down_1))
        conv_down_3 = self._conv_down_3(self._max_pool(conv_down_2))
        conv_bottom = self._conv_bottom(self._max_pool(conv_down_3))
        conv_up_1 = self._conv_up_1(torch.cat((self._up_sample(conv_bottom), conv_down_3), dim=1))
        conv_up_2 = self._conv_up_2(torch.cat((self._up_sample(conv_up_1), conv_down_2), dim=1))
        conv_up_3 = self._conv_up_3(torch.cat((self._up_sample(conv_up_2), conv_down_1), dim=1))
        output = self._conv_heatmap(conv_up_3)
        return output
