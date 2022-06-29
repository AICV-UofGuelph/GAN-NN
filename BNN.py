import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from blitz.modules import BayesianConv2d
# from blitz.utils import variational_estimator










# TODO: Cite BLiTZ
# TODO: Make the gen and disc into inner classes of the Network class?











# Assumptions:
    # Input already processed to desired form

class Network():
    def __init__(self, noise_dim, img_channels, features, device='cpu'):
        self.device = device

        # self.disc = Discriminator(img_channels, features, self.device)
        # self.gen = Generator(noise_dim, img_channels, features, self.device)
        self.build()

        # TODO: Remove the following:
        self.FIXED_NOISE = torch.randn((32, 100, 1, 1), device=self.device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")

    # Initialize the NN models
    def build(self):
        # Init the models' weights
        for m in self.disc.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        for m in self.gen.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

        self.opt_disc= opt.Adam(self.disc.parameters(), lr=1e-7, betas=(0.5, 0.999))
        self.opt_gen= opt.Adam(self.gen.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # TODO: Properly define criterion
        self.criterion = nn.BCELoss()


    # Train the models
    def train(self, dataloader, epochs=1):
        if epochs < 1:
            print('ERROR - Network.train(): arg \'epochs\' must be greater than zero\t\t')
            return

        # TODO: Remove the following:
        BATCH_SIZE = 10
        NOISE_DIMS = (BATCH_SIZE, 100, 1, 1)
        step = 0

        for epoch in range(epochs):
            for batch, real in enumerate(dataloader):
                real = real.to(self.device)
                noise = torch.randn(NOISE_DIMS, device=self.device)

                fake = self.gen(noise)

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                disc_real = self.disc(real.float()).reshape(-1)
                loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.disc(fake.detach()).reshape(-1)
                loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                self.disc.zero_grad()
                loss_disc.backward()
                self.opt_disc.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                output = self.disc(fake).reshape(-1)
                loss_gen = self.criterion(output, torch.ones_like(output))
                self.gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                # Print losses occasionally and print to tensorboard
                if batch % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}] Batch {batch}/{len(dataloader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                    )

                    with torch.no_grad():
                        # fake = self.gen(self.FIXED_NOISE)
                        # take out (up to) 32 examples
                        img_grid_real = torchvision.utils.make_grid(
                            real[:BATCH_SIZE], normalize=True
                        )
                        img_grid_fake = torchvision.utils.make_grid(
                            fake[:BATCH_SIZE], normalize=True
                        )

                        self.writer_real.add_image("Real", img_grid_real, global_step=step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    step += 1



    # Make predictions without updating the models' params
    def generate_path(self, noise):
        with torch.no_grad():
            return self.gen(noise)

    def evaluate_path(self, path):
        with torch.no_grad():
            return self.disc(path)


# @variational_estimator
class Discriminator(nn.Module):
    def __init__(self, img_channels, features, device='cpu'):
        super(Discriminator, self).__init__()

        # FEATURES = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(img_channels, features, kernel_size=4, stride=2, padding=1, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 2, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 4, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 8, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0, device=device), # convert to single channel
            nn.AdaptiveAvgPool2d(1),    # pool the matrix into a single value for sigmoid
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)

        return y


# @variational_estimator
class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels, features, device='cpu'):
        super(Generator, self).__init__()

        # FEATURES = 64

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, features * 16, kernel_size=4, stride=1, padding=0, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 8, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 8, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 4, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False, device=device),
            nn.BatchNorm2d(features * 2, device=device),
            nn.LeakyReLU(0.2)
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, img_channels, kernel_size=4, stride=2, padding=1, device=device),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
    
        return y