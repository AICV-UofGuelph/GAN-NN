import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, f, k, s, p, device='cpu'):
        super(Critic, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(f[0], f[1], k[0], s[0], p[0], device=device, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f[1], f[2], k[1], s[1], p[1], device=device, bias=False),
            nn.InstanceNorm2d(f[2], affine=True, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(f[2], f[3], k[2], s[2], p[2], device=device, bias=False),
            nn.InstanceNorm2d(f[3], affine=True, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(f[3], f[4], k[3], s[3], p[3], device=device, bias=False),
            nn.InstanceNorm2d(f[4], affine=True, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(f[4], f[5], k[4], s[4], p[4], device=device, bias=False)
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        
        return y


# 3 input channels (noise, map, initial path)
class Generator(nn.Module):
    def __init__(self, f, k, s, p, device='cpu'):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(f[0], f[1], k[0], s[0], p[0], device=device),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f[1], f[2], k[1], s[1], p[1], device=device),
            nn.InstanceNorm2d(f[2], affine=True, device=device),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn. Conv2d(f[2], f[3], k[2], s[2], p[2], device=device),
            nn.InstanceNorm2d(f[3], affine=True, device=device),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn. Conv2d(f[3], f[4], k[3], s[3], p[3], device=device),
            nn.InstanceNorm2d(f[4], affine=True, device=device),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn. Conv2d(f[4], f[5], k[4], s[4], p[4], device=device),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn. ConvTranspose2d(f[5], f[6], k[5], s[5], p[5], device=device),
            nn.InstanceNorm2d(f[6], affine=True, device=device),
            nn.ReLU()
        )

        self.block7 = nn.Sequential(
            nn. ConvTranspose2d(f[6], f[7], k[6], s[6], p[6], device=device),
            nn.InstanceNorm2d(f[7], affine=True, device=device),
            nn.ReLU()
        )

        self.block8 = nn.Sequential(
            nn. ConvTranspose2d(f[7], f[8], k[7], s[7], p[7], device=device),
            nn.InstanceNorm2d(f[8], affine=True, device=device),
            nn.ReLU()
        )

        self.block9 = nn.Sequential(
            nn. ConvTranspose2d(f[8], f[9], k[8], s[8], p[8], device=device),
            nn.InstanceNorm2d(f[9], affine=True, device=device),
            nn.ReLU()
        )
        
        self.block10 = nn.Sequential(
            nn. ConvTranspose2d(f[9], f[10], k[9], s[9], p[9], device=device),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.block8(y)
        y = self.block9(y)
        y = self.block10(y)

        y = self._round(y)
        return y
    
    def _round(self, mat):
        # TODO: cite something? (this function is based off of Thor's code)
        mat_hard = torch.round(mat)
        mat = (mat_hard - mat.data) + mat

        return mat