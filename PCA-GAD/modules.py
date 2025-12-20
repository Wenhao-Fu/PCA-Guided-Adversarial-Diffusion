import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            SpectralNorm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = Self_Attn(64)
        self.down2 = Down(64, 128)
        self.sa2 = Self_Attn(128)
        self.down3 = Down(128, 128)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = Self_Attn(64)
        self.up2 = Up(128, 32)
        self.sa5 = Self_Attn(32)
        self.up3 = Up(64, 32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)  # (N, 32, 64, 64)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x2, _ = self.sa1(x2)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x3, _ = self.sa2(x3)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        # x4, _ = self.sa3(x4)
        # print(x4.shape)

        x4 = self.bot1(x4)
        # print(x4.shape)
        x4 = self.bot2(x4)
        # print(x4.shape)

        x = self.up1(x4, x3)
        # print('-----')
        # print(x.shape)
        x, _ = self.sa4(x)
        # print(x.shape)
        x = self.up2(x, x2)
        # print(x.shape)
        x, _ = self.sa5(x)
        # print(x.shape)
        x = self.up3(x, x1)
        # print(x.shape)
        output = self.outc(x)
        # print(output.shape)
        return output


# net = UNet(device="cpu")
# print(sum([p.numel() for p in net.parameters()]))
# x = torch.randn(16, 1, 64, 64)
# output = net(x)
# print(output.shape)


class Generator(nn.Module):
    """Generator."""

    def __init__(self):
        super(Generator, self).__init__()
        self.Unet = UNet()
        self.activation = nn.Tanh()

    def forward(self, z, mean, phi):
        # PCA reconstruction
        x_in = (phi @ z.T + mean).T
        # Standardization
        x_in = x_in * 2 - 1
        # print(x_in.shape)
        x_in = x_in.view(z.size(0), 1, 64, 64)
        # print(x_in.shape)
        out = self.Unet(x_in)
        # print(out.shape)
        out = self.activation(out)

        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256)
        self.attn2 = Self_Attn(512)

    def forward(self, x):
        out = self.l1(x)
        # print(out.shape)
        out = self.l2(out)
        # print(out.shape)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        # print(out.shape)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        # print(out.shape)
        out = self.last(out)
        # print(out.shape)

        return out.squeeze(), p1, p2


if __name__ == '__main__':
    # net = UNet(device="cpu")
    # from torchviz import make_dot
    Gen = Generator()
    print(sum([p.numel() for p in Gen.parameters()]))
    z = torch.randn(8, 200)
    phi = torch.randn(4096, 200)
    mean = torch.randn(4096, 1)
    result = Gen(z, mean, phi)
    print(result.shape)

    x = torch.rand(8, 1, 64, 64)
    Dis = Discriminator()
    print(sum([p.numel() for p in Dis.parameters()]))
    result = Dis(x)
    print(result[0].shape, result[1].shape)

