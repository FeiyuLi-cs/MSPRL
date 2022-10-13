from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSPRL(nn.Module):
    def __init__(self, image_channel, num_blocks=[8, 8, 8, 8]):
        super(MSPRL, self).__init__()
        dim = 48
        # ------------------------------ Content Aggregation ------------------------------
        self.conv1 = nn.Conv2d(image_channel, dim, kernel_size=3, stride=1, padding=1)

        self.ResBlockGroup1_1 = ResBlockGroup(dim, k=3, num_res=num_blocks[0])
        self.down1_2 = Downsample(dim)

        self.SFE1 = SFE(image_channel, channel=dim * 2)
        self.ResBlockGroup2_1 = ResBlockGroup(dim * 2, k=3, num_res=num_blocks[1])
        self.down2_3 = Downsample(dim * 2)

        self.SFE2 = SFE(image_channel, channel=dim * 4)
        self.ResBlockGroup3 = ResBlockGroup(dim * 4, k=3, num_res=num_blocks[2])

        self.up3_2 = Upsample(dim * 4)
        self.FF1 = FF(dim * 2 * 2, dim * 2)
        self.ResBlockGroup2_2 = ResBlockGroup(dim * 2, k=3, num_res=num_blocks[1])

        self.up2_1 = Upsample(dim * 2)
        self.FF2 = FF(dim * 2, dim)
        self.ResBlockGroup1_2 = ResBlockGroup(dim, k=3, num_res=num_blocks[0])
        self.out = nn.Conv2d(dim, image_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        # ------------------------------ Content Aggregation ------------------------------
        out = self.conv1(x)

        out = self.ResBlockGroup1_1(out)
        res1 = out
        out = self.down1_2(out)

        out = self.SFE1(x_2, out)
        out = self.ResBlockGroup2_1(out)
        res2 = out
        out = self.down2_3(out)

        out = self.SFE2(x_4, out)
        out = self.ResBlockGroup3(out)

        out = self.up3_2(out)
        out = self.FF1(out, res2)
        out = self.ResBlockGroup2_2(out)

        out = self.up2_1(out)
        out = self.FF2(out, res1)
        out = self.ResBlockGroup1_2(out)

        out = self.out(out)
        out = out + x

        return out


class SFE(nn.Module):
    def __init__(self, image_channel, channel):
        super(SFE, self).__init__()
        planes = image_channel + channel
        self.enconv = nn.Sequential(
            nn.Conv2d(image_channel, channel, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, stride=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(planes, channel, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x, y):
        res = self.enconv(x)
        res = res * y
        res = torch.cat([x, res], dim=1)
        res = self.conv(res)
        res = res + y
        return res


class FF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class ResBlockGroup(nn.Module):
    def __init__(self, channel, k, num_res):
        super(ResBlockGroup, self).__init__()

        layers = [ResBlock(channel, channel, k=k, planes=channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, planes):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, planes, kernel_size=k, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, out_channel, kernel_size=k, padding=1, stride=1)
        )

    def forward(self, x):
        res = self.conv(x)
        res = res + x
        return res


# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


if __name__ == '__main__':
    net = MSPRL(image_channel=1)
    print(sum([param.nelement() for param in net.parameters()]))
    # print(net)
    y = net(torch.randn(16, 1, 128, 128))
    print(y.size())
