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
    
class ResNet(nn.Module):
    '''
    Residual network

    Arg(s):
		n_layers : int
			number of layers (18 or 34)
        n_input_feature : int
            number of input features
        batch_norm : bool
            whether or not to use batch normalization
    '''
    
    layer_counts = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
    }
    
    def __init__(self, n_layers, n_input_feature, batch_norm=False):
        super(ResNet, self).__init__()

        if batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(n_input_feature, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(n_input_feature, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.lc = self.layers_counts[n_layers]
  
        self.layer2 = self.make_layer(64, 64, self.lc[0], batch_norm)
        self.layer3 = self.make_layer(64, 128, self.lc[1], batch_norm)
        self.layer4 = self.make_layer(128, 256, self.lc[2], batch_norm)
        self.layer5 = self.make_layer(256, 512, self.lc[3], batch_norm)

		# make output sizes match
        self.layer6 = nn.Sequential(
            ResNetBlock(512, 256, 2, batch_norm),
            ResNetBlock(256, 128, 2, batch_norm),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.output = nn.Linear(512, n_output)
    
    def make_layer(self, input, output, count, batch_norm):
        '''
        Construct block of layers for ResNet

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
        if input == output:
            layers.append(ResNetBlock(input, output, 1, batch_norm))
        else:
            layers.append(ResNetBlock(input, output, 2, batch_norm))
        count -= 1
        
        while count > 0:
            layers.append(ResNetBlock(output, output, 1, batch_norm))
            count -= 1
            
        return nn.Sequential(*layers)
            

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

