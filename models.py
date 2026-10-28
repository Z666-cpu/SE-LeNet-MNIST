import torch
import torch.nn as nn

# SE通道注意力模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 基线模型 LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 改进模型 SE-LeNet（仅通道注意力，用于消融实验）
class SELenet(nn.Module):
    def __init__(self, num_classes=10, reduction=2):
        super(SELenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        # 在深层卷积层(120通道)后嵌入SE通道注意力模块
        self.se = SEModule(120, reduction)
        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 改进模型 BN-LeNet（仅批归一化，用于消融实验）
class BNLenet(nn.Module):
    def __init__(self, num_classes=10):
        super(BNLenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 改进模型 SE-BN-LeNet（批归一化 + 通道注意力，最终改进方案）
class SEBNLenet(nn.Module):
    def __init__(self, num_classes=10, reduction=2):
        super(SEBNLenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
        # 在深层卷积层(120通道)后嵌入SE通道注意力模块
        self.se = SEModule(120, reduction)
        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
