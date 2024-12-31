import torch
from torch import nn
from torch.nn import functional as F
import torch as t




class Segnet(nn.Module):
    def __init__(self, input_nc, output_nc):  # 将 init 改为 __init__
        super(Segnet, self).__init__()
        # Encoder
        self.conv11 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1)  ##[4,256,256]-->[64,256,256]
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)

        # Decoder
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, output_nc, kernel_size=3, padding=1)
    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)), inplace=True)
        x12 = F.relu(self.bn12(self.conv12(x11)), inplace=True)
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)), inplace=True)
        x22 = F.relu(self.bn22(self.conv22(x21)), inplace=True)
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)), inplace=True)
        x32 = F.relu(self.bn32(self.conv32(x31)), inplace=True)
        x33 = F.relu(self.bn33(self.conv33(x32)), inplace=True)
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)), inplace=True)
        x42 = F.relu(self.bn42(self.conv42(x41)), inplace=True)
        x43 = F.relu(self.bn43(self.conv43(x42)), inplace=True)
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)), inplace=True)
        x52 = F.relu(self.bn52(self.conv52(x51)), inplace=True)
        x53 = F.relu(self.bn53(self.conv53(x52)), inplace=True)
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)), inplace=True)
        x52d = F.relu(self.bn52d(self.conv52d(x53d)), inplace=True)
        x51d = F.relu(self.bn51d(self.conv51d(x52d)), inplace=True)

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)), inplace=True)
        x42d = F.relu(self.bn42d(self.conv42d(x43d)), inplace=True)
        x41d = F.relu(self.bn41d(self.conv41d(x42d)), inplace=True)

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)), inplace=True)
        x32d = F.relu(self.bn32d(self.conv32d(x33d)), inplace=True)
        x31d = F.relu(self.bn31d(self.conv31d(x32d)), inplace=True)

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)), inplace=True)
        x21d = F.relu(self.bn21d(self.conv21d(x22d)), inplace=True)

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)), inplace=True)
        x11d = self.conv11d(x12d)
        # output = t.sigmoid(x11d)

        return x11d


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_num_filters=16):
        super(UNet, self).__init__()
        # Down-sampling path (contracting path)
        self.conv1 = nn.Conv2d(in_channels, base_num_filters, kernel_size=3,
                               padding=1)  # (N, 3, 128, 128)->(N, 16, 128, 128)
        # max pooling
        self.conv2 = nn.Conv2d(base_num_filters, base_num_filters * 2, kernel_size=3,
                               padding=1)  # (N, 16, 64, 64)->(N, 32, 64, 64)
        # max pooling
        self.conv3 = nn.Conv2d(base_num_filters * 2, base_num_filters * 4, kernel_size=3,
                               padding=1)  # (N, 32, 32, 32)->(N, 128, 32, 32)
        # max pooling
        self.conv4 = nn.Conv2d(base_num_filters * 4, base_num_filters * 8, kernel_size=3,
                               padding=1)  # (N, 128, 16, 16)->(N, 256, 16, 16)
        # max pooling
        self.conv5 = nn.Conv2d(base_num_filters * 8, base_num_filters * 16, kernel_size=3,
                               padding=1)  # (N, 256, 8, 8)->(N, 512, 8, 8)

        # Up-sampling path (expansive path)
        self.upconv4 = nn.ConvTranspose2d(base_num_filters * 16, base_num_filters * 8, kernel_size=2,
                                          stride=2)  # (N, 512, 8, 8)->(N, 256, 16, 16)
        # concatenate (upconv4, conv4)
        self.conv6 = nn.Conv2d(base_num_filters * 16, base_num_filters * 8, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(base_num_filters * 8, base_num_filters * 4, kernel_size=2, stride=2)
        # concatenate (upconv3, conv3)
        self.conv7 = nn.Conv2d(base_num_filters * 8, base_num_filters * 4, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(base_num_filters * 4, base_num_filters * 2, kernel_size=2, stride=2)
        # concatenate (upconv2, conv2)
        self.conv8 = nn.Conv2d(base_num_filters * 4, base_num_filters * 2, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(base_num_filters * 2, base_num_filters, kernel_size=2, stride=2)
        # concatenate (upconv1, conv1)
        self.conv9 = nn.Conv2d(base_num_filters * 2, base_num_filters, kernel_size=3, padding=1)

        # Final layer
        self.conv10 = nn.Conv2d(base_num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Down-sampling path
        c1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)

        c2 = F.relu(self.conv2(p1))
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = F.relu(self.conv3(p2))
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)

        c4 = F.relu(self.conv4(p3))
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)

        c5 = F.relu(self.conv5(p4))

        # Up-sampling path
        up4 = self.upconv4(c5)
        up4 = torch.cat([up4, c4], dim=1)
        c6 = F.relu(self.conv6(up4))

        up3 = self.upconv3(c6)
        up3 = torch.cat([up3, c3], dim=1)
        c7 = F.relu(self.conv7(up3))

        up2 = self.upconv2(c7)
        up2 = torch.cat([up2, c2], dim=1)
        c8 = F.relu(self.conv8(up2))

        up1 = self.upconv1(c8)
        up1 = torch.cat([up1, c1], dim=1)
        c9 = F.relu(self.conv9(up1))

        # Final layer
        out = self.conv10(c9)
        return out
