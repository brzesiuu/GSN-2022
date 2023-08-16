import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from decorators.conversion_decorators import keypoints_2d


class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseEstimationModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.backbone = models.resnet18(ResNet18_Weights.IMAGENET1K_V1)

        # Remove fully connected layers and add a new one for regression
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self._deconv_1 = self._make_deconv(512, 256, 4)
        self._deconv_2 = self._make_deconv(256, 256, 4)
        self._deconv_3 = self._make_deconv(256, 256, 2)

        # Add a new convolutional layer to predict heatmaps
        self.heatmap_layer = nn.Conv2d(256, num_keypoints, kernel_size=1, padding='same')

    def _make_deconv(self, in_filters, out_filters, scale):
        return nn.Sequential(nn.ConvTranspose2d(in_filters, out_filters, scale, scale, bias=False),
                             nn.BatchNorm2d(out_filters),
                             nn.ReLU())

    @keypoints_2d
    def forward(self, x):
        features = self.backbone(x)
        deconv = self._deconv_1(features)
        deconv = self._deconv_2(deconv)
        deconv = self._deconv_3(deconv)
        heatmaps = self.heatmap_layer(deconv)
        return {'heatmaps': heatmaps}
