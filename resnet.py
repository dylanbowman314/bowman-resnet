import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    '''ResNet is a convolutional neural network that uses residual connections.'''
    def __init__(self, n):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Dimensions here are hard-coded for CIFAR-10.
        self.fb1 = FeatureBlock(16,16,n)
        self.fb2 = FeatureBlock(16,32,n)
        self.fb3 = FeatureBlock(32,64,n)

        self.pool = nn.AvgPool2d(kernel_size=8)
        self.fc1 = nn.Linear(64,10)

    def forward(self,img):
        img = img - torch.mean(img)

        img = self.conv1(img)

        img = self.fb1(img)
        img = self.fb2(img)
        img = self.fb3(img)

        img = self.pool(img)

        img = torch.squeeze(img)

        img = self.fc1(img)
        img = F.softmax(img,1)

        return img
    
class FeatureBlock(nn.Module):
    '''A block containing n residual convolutional blocks. The first residual block decreases the size of each feature map and doubles the number of feature maps.'''
    def __init__(self, in_channels, out_channels, n):
        super(FeatureBlock, self).__init__()
        
        self.layers = nn.ModuleList()

        self.layers.append(ResBlock(in_channels, out_channels))

        for _ in range(1,n):
            self.layers.append(ResBlock(out_channels, out_channels))
    
    def forward(self, img):
        for l in self.layers:
            img = l(img)
        return img
    
class ResBlock(nn.Module):
    '''Simple residual block which performs two convolutions with batch normalization and then applies the residual connection.'''
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.disc = in_channels == out_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1 if self.disc else 2,
            padding=1,
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.fix_channels = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, img):
        old_img = img

        img = self.conv1(img)
        img = self.bn1(img)
        img = F.relu(img)

        img = self.conv2(img)
        img = self.bn2(img)

        if self.disc:
            img += old_img
        else:
            k = self.fix_channels(old_img)
            img += k
        
        img = F.relu(img)

        return img