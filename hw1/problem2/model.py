"""PreActResNet-18 adapted from the course multi_norm_robustness.ipynb starter.

https://github.com/nalinwadhwa02/CCS521FA26HW/blob/main/hw1/multi_norm_robustness.ipynb
"""

import torch
from torch import nn
from torch.nn import functional as F


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                          bias=False)
            )

    def forward(self, x):
        out = F.softplus(self.bn1(x), beta=1)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.softplus(self.bn2(out), beta=1))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, n_classes)

    def _make_layer(self, planes, count, stride):
        layers = []
        for step_stride in [stride] + [1] * (count - 1):
            layers.append(PreActBlock(self.in_planes, planes, step_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  # The starter checkpoint uses normal='none'.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        return self.linear(out.flatten(1))
