from torch import nn


class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualModule, self).__init__()
        self._conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels / 2), kernel_size=1, padding='same',
                      stride=1),
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.ReLU())
        self._conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels / 2), out_channels=int(out_channels / 2), kernel_size=3,
                      padding='same',
                      stride=1),
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.ReLU()
        )
        self._conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1,
                      padding='same',
                      stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self._conv_1(x)
        out = self._conv_2(out)
        out = self._conv_3(out)
        return x + out


class HourglassModule(nn.Module):
    def __init__(self, num_features=256):
        super(HourglassModule, self).__init__()

        self._max_pool = nn.MaxPool2d(2, 2)
        self._upsample = nn.Upsample(scale_factor=2)

        self._conv_1_up = ResidualModule(num_features, num_features)
        self._conv_1_middle = ResidualModule(num_features, num_features)
        self._conv_1_down = ResidualModule(num_features, num_features)

        self._conv_2_up = ResidualModule(num_features, num_features)
        self._conv_2_middle = ResidualModule(num_features, num_features)
        self._conv_2_down = ResidualModule(num_features, num_features)

        self._conv_3_up = ResidualModule(num_features, num_features)
        self._conv_3_middle = ResidualModule(num_features, num_features)
        self._conv_3_down = ResidualModule(num_features, num_features)

        self._conv_4_up = ResidualModule(num_features, num_features)
        self._conv_4_middle = ResidualModule(num_features, num_features)
        self._conv_4_down = ResidualModule(num_features, num_features)

        self._conv_5_1 = ResidualModule(num_features, num_features)
        self._conv_5_2 = ResidualModule(num_features, num_features)
        self._conv_5_3 = ResidualModule(num_features, num_features)

    def forward(self, x):
        conv_1 = self._conv_1_down(x)
        conv_2 = self._conv_2_down(self._max_pool(conv_1))
        conv_3 = self._conv_3_down(self._max_pool(conv_2))
        conv_4 = self._conv_4_down(self._max_pool(conv_3))
        conv_5 = self._conv_5_3(self._conv_5_2(self._conv_5_1(self._max_pool(conv_4))))

        x = self._upsample(conv_5)
        x = self._upsample(self._conv_4_up(self._conv_4_middle(conv_4) + x))
        x = self._upsample(self._conv_3_up(self._conv_3_middle(conv_3) + x))
        x = self._upsample(self._conv_2_up(self._conv_2_middle(conv_2) + x))
        return self._conv_1_up(self._conv_1_middle(conv_1) + x)
