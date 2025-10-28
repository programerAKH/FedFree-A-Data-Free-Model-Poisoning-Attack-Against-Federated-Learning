"""
定义神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFmnist(nn.Module):
    def __init__(self):
        super(CNNFmnist,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FCNetMNIST(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, dropout_prob=0.5):
        super(FCNetMNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入图像 (batch_size, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


"""class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()

        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输出: 32x32x32
            nn.GroupNorm(4, 32),  # 使用更少的分组
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16x32

            nn.Conv2d(32, 96, kernel_size=3, padding=1),  # 输出: 16x16x96
            nn.GroupNorm(4, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 8x8x96

            # 移除了一个卷积层
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # 输出: 8x8x128
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 输出: 8x8x128
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 4x4x128
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 输出: 2x2x128 → 512元素

        # 简化全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 降低dropout比例
            nn.Linear(128 * 2 * 2, 256),  # 512 → 256
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),  # 直接输出到类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x"""


class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()

        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 输出通道减半
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16x16

            # 移除了GroupNorm层
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输出通道减半
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 8x8x32

            # 移除了一个卷积层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输出: 8x8x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 4x4x64
        )

        # 移除了自适应池化层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 64),  # 特征维度大幅降低
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)  # 直接输出分类结果
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x


# Basic blocks for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # stages
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet34_CIFAR(num_classes=100):
    return ResNet_CIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
