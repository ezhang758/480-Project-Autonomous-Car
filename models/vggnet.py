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

class VGGNet(nn.Module):
    '''
    VGG network

    Arg(s):
        n_input_feature : int
            number of input features
        n_layers : int
			number of layers (11, 13, 16, or 19)
        batch_norm : bool
            whether or not to use batch normalization
    '''
    layer_count = {
		11 : [1, 1, 2, 2, 2],
		13 : [2, 2, 2, 2, 2],
		16 : [2, 2, 3, 3, 3],
		19 : [2, 2, 4, 4, 4],
	}

    def __init__(self, n_input_feature, n_layers, batch_norm=False):
        super(VGGNet, self).__init__()
        
        self.lc = self.layer_count[n_layers]
        
        self.layer1 = self.make_layer(n_input_feature, 64, self.lc[0], batch_norm)
        self.layer2 = self.make_layer(64, 128, self.lc[1], batch_norm)
        self.layer3 = self.make_layer(128, 256, self.lc[2], batch_norm)
        self.layer4 = self.make_layer(256, 512, self.lc[3], batch_norm)
        self.layer5 = self.make_layer(512, 512, self.lc[4], batch_norm)
        
        # added layer to have feature the same size as mask
        self.layer6 = nn.Sequential(
            VGGNetBlock(512, 256, batch_norm),
            VGGNetBlock(256, 128, batch_norm),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
    
    def make_layer(self, n_input, n_output, count, batch_norm):
        '''
        Construct block of layers for VGGNet

        Arg(s):
            input : int
                input dimensions
            output : int
                output dimensions
            count : int
                number of layers in block
            batch_norm : bool
                whether or not to use batch normalization
        Returns:
            nn.Sequential
                layer
        '''
        layers = []
        
        layers.append(VGGNetBlock(n_input, n_output, batch_norm))
        count -= 1
        
        while count > 0:
            layers.append(VGGNetBlock(n_output, n_output, batch_norm))
            count -= 1
        
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        return nn.Sequential(*layers)

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
