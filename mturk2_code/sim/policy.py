import torch
from torch import nn
from torch import tensor
from torch.utils.data import Dataset, DataLoader


class QNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=8) # downsample to 50x50x3
        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=10, stride=1, padding=1)  # 50x50x40
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)  # 25x25x10
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=10, out_channels=20, stride=1, padding=1)  # 25x25x20
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1)  # 9x9x20
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(kernel_size=3, in_channels=20, out_channels=10, stride=1, padding=1)  # 9x9x4
        self.pool3 = nn.MaxPool2d(kernel_size=3, padding=0)  # 3x3x4
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(kernel_size=3, padding=0, in_channels=10, out_channels=4)  #
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.downsample(x)
        h = self.conv1(h)
        h = self.pool1(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.conv3(h)
        h = self.pool3(h)
        h = self.bn3(h)
        h = self.activation(h)
        h = self.conv4(h)
        y = self.softmax(h.reshape(-1, 4)).clone()
        return y



