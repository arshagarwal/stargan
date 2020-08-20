import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
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
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class Generator2(torch.nn.Module):
    """Given latent vector, produces Image of the shape (128,128,3)
      latent vector shape: [batch_size,z+ny]
      where
      z: is the number of parameters into which which image is encoded using Encoder_z (100)
      ny:number of attributes in an image
      """
    def __init__(self,ny,l):
        """

        :param l: number of units in the latent vector
        :param ny: number of categories
        """
        super(Generator2,self).__init__();
        self.l=l

        self.conv = torch.nn.ConvTranspose2d(self.l,1024,4,1,0,bias=False)


        # embedding layer
        self.embedding =torch.nn.Embedding(ny,50)
        self.dense2 = torch.nn.Linear(50,4*4,bias=False)



        # conv layers
        self.conv0 = torch.nn.ConvTranspose2d(1025,512,4,2,1,bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(512,256,4,2,1,bias=False)

        self.conv2 = torch.nn.ConvTranspose2d(256,128,4,2,1,bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(128,64,4,2,1,bias=False)
        self.conv4 = torch.nn.ConvTranspose2d(64,3,4,2,1,bias=False)

        #tanh activation
        self.tanh=torch.nn.Tanh()

        # batch_norm layers
        self.batch_norm0=nn.InstanceNorm2d(512, affine=True, track_running_stats=True)
        self.batch_norm1=nn.InstanceNorm2d(256, affine=True, track_running_stats=True)
        self.batch_norm2 = nn.InstanceNorm2d(128, affine=True, track_running_stats=True)
        self.batch_norm3 = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
        self.conv_batch_norm=nn.InstanceNorm2d(1024, affine=True, track_running_stats=True)

        # leaky relu layers
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.relu3 = torch.nn.LeakyReLU(0.2)
        self.relu4 = torch.nn.LeakyReLU(0.2)
        self.relu5 = torch.nn.LeakyReLU(0.2)


    def forward(self,input,labels):
        """
        Generates synthetic images of the shape [batch_size,128,128,3]
        :param input: Latent vector + attributes of the shape [batch_size,z+l]
        :return: synthetic images of the shape [batch_size,128,128,3]
        """
        # creating the embedding
        embedding=self.embedding(labels)
        assert embedding.shape == (input.shape[0], 50), " check generator embedding shape {} ".format(embedding.shape)
        embedding=self.dense2(embedding)
        embedding=torch.reshape(embedding,(input.shape[0],1,4,4))

        input = torch.reshape(input, (input.shape[0],self.l,1,1) )
        x=self.conv(input)
        x = self.conv_batch_norm(x)
        x=self.relu1(x)


        assert x.shape==(input.shape[0],1024,4,4), 'check generators first conv layer'

        # concatenating embedding
        x=torch.cat((x,embedding),axis=1);

        assert x.shape == (input.shape[0],1025,4,4), "check embedding concatenation"

        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu2(x)

        x=self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu3(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu4(x)


        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu5(x)


        x = self.conv4(x)

        return x

