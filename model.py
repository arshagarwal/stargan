import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as SPN

class Attention(nn.Module):
    """Attention module as per SA-GAN official implementation"""
    def __init__(self,in_channels):
        super(Attention, self).__init__()
        self.in_channels=in_channels

        self.sigma=torch.nn.parameter.Parameter(torch.tensor(0.0,requires_grad=True))

        self.maxPool=nn.MaxPool2d(kernel_size=2)

        self.theta=SPN(nn.Conv2d(self.in_channels,self.in_channels//8,1,1,bias=False))
        self.phi=SPN(nn.Conv2d(self.in_channels,self.in_channels//8,1,1,bias=False))
        self.g=SPN(nn.Conv2d(self.in_channels,self.in_channels//2,1,1,bias=False))
        self.final=SPN(nn.Conv2d(self.in_channels//2,self.in_channels,1,1,bias=False))

    def forward(self,x):
        (B,C,H,W)=x.shape
        theta = self.theta(x)
        theta = torch.reshape(theta,(theta.shape[0],theta.shape[1],-1)).permute(0,2,1) # shape (B,H*W,C/8)
        assert theta.shape == (B, H*W,self.in_channels // 8), "check theta shape Attention Module"

        phi = self.maxPool(self.phi(x))
        phi = torch.reshape(phi,(x.shape[0],self.in_channels//8,-1)) # shape(B,C/8,H*W/4)

        assert phi.shape == (B,self.in_channels//8,H*W/4), "check phi shape Attention Module"

        attn = torch.bmm(theta,phi)
        attn = torch.softmax(attn,dim=-1) # shape(B,H*W,H*W/4)

        g = self.maxPool(self.g(x))
        g=torch.reshape(g,(x.shape[0],self.in_channels//2,-1)) # shape=(B,C/2,H*w/4)
        g = g.permute(0,2,1) # shape=(B,H*W/4,C/2)

        attn_g = torch.bmm(attn,g).permute(0,2,1) # shape=(B,C/2,H*W)
        attn_g=torch.reshape(attn_g,(B,self.in_channels//2,H,W))
        attn_g = self.final(attn_g)

        assert attn_g.shape == x.shape,"check Attention Module"

        assert self.sigma.device==attn_g.device, "check device allocation in Attention Module"
        assert x.device==self.sigma.device, "x.device {} sigma.device {}".format(x.device,self.sigma.device)

        res=self.sigma*attn_g + x


        assert res.shape==(B,C,H,W), "check Attention Module"

        return res


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            SPN(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            SPN(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(SPN(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(SPN(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            if i==1:
                layers.append(Attention(curr_dim))


        # Up-sampling layers.
        for i in range(2):
            layers.append(SPN(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(SPN(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SPN(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
            if i == 1:
                layers.append(Attention(curr_dim))

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

