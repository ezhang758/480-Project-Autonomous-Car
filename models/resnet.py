import numpy as np
import torch.nn as nn

class ResNetBlock(nn.Module):
    '''
    Residual network block

    Arg(s):
        n_input_feature : int
            number of input feature channels
        n_output_feature : int
            number of output feature channels i.e. number of filters to use
        stride : int
            stride of convolution
        batch_norm : bool
            whether or not to use batch normalization
    '''

    def __init__(self, n_input_feature, n_output_feature, stride, batch_norm=False):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            n_input_feature,
            n_output_feature,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            n_output_feature,
            n_output_feature,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if batch_norm:
            self.bn = nn.BatchNorm2d(n_output_feature)
        else:
            self.bn = None

        self.relu = nn.ReLU()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                in_channels=n_input_feature,
                out_channels=n_output_feature,
                kernel_size=1,
                stride=stride,
                padding=0,
            )


    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        identity = x
        if self.downsample:
            identity = self.downsample(identity)
            if self.bn:
                identity = self.bn(identity)

        x = self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn(x)
        x += identity
        output_features = self.relu(x)

        return output_features
    
class ResNet18(nn.Module):
    '''
    Residual network with 18 layers (ResNet18)

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
        batch_norm : bool
            whether or not to use batch normalization
    '''
    def __init__(self, n_input_feature, n_output, batch_norm=False):
        super(ResNet18, self).__init__()

        if batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(n_input_feature, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(n_input_feature, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )

        self.layer2 = nn.Sequential(
            ResNetBlock(64, 64, 1, batch_norm),
            ResNetBlock(64, 64, 1, batch_norm),
        )

        self.layer3 = nn.Sequential(
            ResNetBlock(64, 128, 2, batch_norm),
            ResNetBlock(128, 128, 1, batch_norm),
        )

        self.layer4 = nn.Sequential(
            ResNetBlock(128, 256, 2, batch_norm),
            ResNetBlock(256, 256, 1, batch_norm),
        )

        self.layer5 = nn.Sequential(
            ResNetBlock(256, 512, 2, batch_norm),
            ResNetBlock(512, 512, 1, batch_norm),
        )

        self.layer6 = nn.Sequential(
            ResNetBlock(512, 256, 2, batch_norm),
            ResNetBlock(256, 128, 2, batch_norm),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.output = nn.Linear(512, n_output)


    def forward(self, x):
        '''
        Forward pass through ResNet

        Arg(s):
            x : torch.Tensor[float32]
                tensor of N x d
        Returns:
            torch.Tensor[float32]
                tensor of n_output predicted class
        '''

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        # output_logits = self.output(x)

        return x

