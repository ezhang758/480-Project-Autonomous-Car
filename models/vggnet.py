import numpy as np
import torch.nn as nn

class VGGNetBlock(nn.Module):
    '''
    Residual network block

    Arg(s):
        n_input_feature : int
            number of input feature channels
        n_output_feature : int
            number of output feature channels i.e. number of filters to use
        batch_norm : bool
            whether or not to use batch normalization
            default: False
    '''

    def __init__(self, n_input_feature, n_output_feature, batch_norm=False):
        super(VGGNetBlock, self).__init__()

        self.conv = nn.Conv2d(n_input_feature, n_output_feature, kernel_size=3, stride=1, padding=1)
        # normalize conv weights
        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.out_channels
        self.conv.weight.data.normal_(0, np.sqrt(2. / n))
        self.conv.bias.data.zero_()

        self.nonlin = nn.ReLU(inplace=True)

        if batch_norm:
            self.bn = nn.BatchNorm2d(n_output_feature)
        else:
            self.bn = None


    def forward(self, x):
        '''
        Forward input x through a VGGNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        output_features = self.nonlin(x)

        return output_features

class VGGNet11(nn.Module):
    '''
    VGG network with 11 layers (VGGNet11)

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
        drop_out : float or None
            probability of an element to be zeroed
            default : None
        batch_norm : bool
            whether or not to use batch normalization
    '''

    def __init__(self, n_input_feature, n_output, batch_norm=False):
        super(VGGNet11, self).__init__()

        self.layer1 = nn.Sequential(
            VGGNetBlock(n_input_feature, 64, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            VGGNetBlock(64, 128, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            VGGNetBlock(128, 256, batch_norm),
            VGGNetBlock(256, 256, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = nn.Sequential(
            VGGNetBlock(256, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = nn.Sequential(
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # added layer to have feature the same size as mask
        self.layer6 = nn.Sequential(
            VGGNetBlock(512, 256, batch_norm),
            VGGNetBlock(256, 128, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        '''
        Forward pass through VGGNet11

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

        return x

class VGGNet19(nn.Module):
    '''
    VGG network with 19 layers (VGGNet19)

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
        drop_out : float or None
            probability of an element to be zeroed
            default : None
        batch_norm : bool
            whether or not to use batch normalization
    '''

    def __init__(self, n_input_feature, n_output, batch_norm=False):
        super(VGGNet19, self).__init__()

        self.layer1 = nn.Sequential(
            VGGNetBlock(n_input_feature, 64, batch_norm),
            VGGNetBlock(64, 64, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            VGGNetBlock(64, 128, batch_norm),
            VGGNetBlock(128, 128, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            VGGNetBlock(128, 256, batch_norm),
            VGGNetBlock(256, 256, batch_norm),
            VGGNetBlock(256, 256, batch_norm),
            VGGNetBlock(256, 256, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = nn.Sequential(
            VGGNetBlock(256, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = nn.Sequential(
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            VGGNetBlock(512, 512, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # added layer to have feature the same size as mask
        self.layer6 = nn.Sequential(
            VGGNetBlock(512, 256, batch_norm),
            VGGNetBlock(256, 128, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        '''
        Forward pass through VGGNet19

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

        return x

