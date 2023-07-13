from torch import nn

from decorators.conversion_decorators import keypoints_2d
from models.hourglass_module import HourglassModule


class StackedHourglass(nn.Module):
    def __init__(self, out_features=21, in_channels=3, hourglass_features=256, num_layers=2):
        super(StackedHourglass, self).__init__()
        self._conv_in = nn.Sequential(nn.BatchNorm2d(in_channels),
                                      nn.Conv2d(in_channels=in_channels, out_channels=hourglass_features, kernel_size=5,
                                                padding='same'),
                                      nn.ReLU())
        self._stacked_hourglass = nn.Sequential(*[HourglassModule(hourglass_features)] * num_layers)
        self._conv_out = nn.Conv2d(in_channels=hourglass_features, out_channels=out_features, padding='same',
                                   kernel_size=5)

    @keypoints_2d
    def forward(self, x):
        x = self._conv_in(x)
        x = self._stacked_hourglass(x)
        output = self._conv_out(x)
        return {'heatmaps': output}
