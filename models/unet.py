class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, model, n_classes):
        super(MyUNet, self).__init__()
        # self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.base_model = ResNet18(3, n_classes)
        if model == "VGGNet11":
            self.base_model = VGGNet11(3, n_classes)
        elif model == "VGGNet19":
            self.base_model = VGGNet19(3, n_classes)
        elif model == "ResNet18":
            self.base_model = ResNet18(3, n_classes)

        self.conv0 = double_conv(5, 64) # changed 5 to 3
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(128 + 1024, 512)
        # self.up1 = up(2304, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)


    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0)) # changed x0 to x
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        # N, C, W, H
        # feature extractor downsample
        # x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8] # centered x
        # feats = self.base_model.extract_features(x_center) # (4, 1280, 10, 24) shape
        feats_var = self.base_model(x)
        feats = feats_var.data
        # print("shape of feats: ", feats.shape)
        # bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        # feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        # mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x