"""
Building a CNN model from scratch
"""
import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(num_features = 32)
        self.relu1 = nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.block1 = Block(in_channels = 32, out_channels = 64)
        self.block2 = Block(in_channels = 64, out_channels = 128)
        self.block3 = Block(in_channels = 128, out_channels = 256)
        self.block4 = Block(in_channels = 256, out_channels = 512)
        self.max_pooling2 = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.max_pooling2(x)

        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        return x
        


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # Skip connection handling
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.shortcut(y)

        x = self.relu2(x)
        return x

if __name__ == "__main__":
    img = torch.rand(1, 3, 224, 224)
    model = CNN(num_classes = 10)
    img = model(img)
    print(img.shape)

