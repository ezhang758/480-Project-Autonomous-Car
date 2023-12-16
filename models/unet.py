import torch
import torch.nn as nn
from models.resnet import ResNet18, ResNet34
from models.vggnet import VGGNet11, VGGNet19
from utils.utils import get_mesh
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2) 

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        Y_off = x2.size()[2] - x1.size()[2]
        X_off = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (X_off // 2, X_off - X_off//2,
                        Y_off // 2, Y_off - Y_off//2))

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, model, n_classes):
        super(UNet, self).__init__()
        if model == "VGGNet-11":
            self.base_model = VGGNet11(3, n_classes)
        elif model == "VGGNet-19":
            self.base_model = VGGNet19(3, n_classes)
        elif model == "ResNet-34":
            self.base_model = ResNet34(3, n_classes)

        self.conv0 = double_conv(5, 64) 
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        self.mp = nn.MaxPool2d(2)
        self.up1 = up(128 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0)) 
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        # N, C, W, H
        # feature extractor downsample
        feats_var = self.base_model(x)
        feats = feats_var.data

        # upsample
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(up, self).__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv = UNetBlock(in_ch, out_ch)

#     def forward(self, x1, x2=None):
#         x1 = self.up(x1)

#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))

#         if x2 is not None:
#             x = torch.cat([x2, x1], dim=1)
#         else:
#             x = x1
#         x = self.conv(x)
#         return x

# class UNetBlock(nn.Module):
#     '''
#     UNet block

#     Arg(s):
#         n_input_feature : int
#             number of input feature channels
#         n_output_feature : int
#             number of output feature channels i.e. number of filters to use
#     '''

#     def __init__(self, n_input_feature, n_output_feature):
#         super(UNetBlock, self).__init__()

#         self.conv1 = nn.Conv2d(
#             n_input_feature,
#             n_output_feature,
#             kernel_size=3,
#             padding=1,
#         )

#         self.conv2 = nn.Conv2d(
#             n_output_feature,
#             n_output_feature,
#             kernel_size=3,
#             padding=1,
#         )

#         self.bn = nn.BatchNorm2d(n_output_feature)
        
#         self.relu = nn.ReLU()


#     def forward(self, x):
#         '''
#         Forward input x through a basic UNet block

#         Arg(s):
#             x : torch.Tensor[float32]
#                 N x C x H x W input tensor
#         Returns:
#             torch.Tensor[float32] : N x K x h x w output tensor
#         '''

#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.relu(x)

#         return x
    
    
# class UNet(nn.Module):
#     def __init__(self, model, n_classes):
#         super(UNet, self).__init__()
#         model_type = model.split("-")
#         if model_type[0] == "VGGNet":
#             # self.base_model = VGGNet(3, int(model_type[1]))
#             # self.base_model = VGGNet(3, 8)
#         elif model_type[0] == "ResNet":
#             self.base_model = ResNet(3, int(model_type[1]))

#         self.conv0 = UNetBlock(5, 64) 
#         self.conv1 = UNetBlock(64, 128)
#         self.conv2 = UNetBlock(128, 512)
#         self.conv3 = UNetBlock(512, 1024)

#         self.mp = nn.MaxPool2d(2)

#         self.up1 = up(128 + 1024, 512)
#         self.up2 = up(512 + 512, 256)
        
#         self.outc = nn.Conv2d(256, n_classes, 1)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
#         x0 = torch.cat([x, mesh1], 1)
#         x1 = self.mp(self.conv0(x0)) 
#         x2 = self.mp(self.conv1(x1))
#         x3 = self.mp(self.conv2(x2))
#         x4 = self.mp(self.conv3(x3))

#         feats_var = self.base_model(x)
#         feats = feats_var.data

#         x = self.up1(feats, x4)
#         x = self.up2(x, x3)
#         x = self.outc(x)
#         return x