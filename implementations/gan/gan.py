import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )


    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class GAN():
    def __init__(self, dataloader, latent_dim, cuda=torch.cuda.is_available()):
        
        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.cuda = cuda
        
        #get shape of image with a simple trick
        for X, _ in self.dataloader:
            self.img_shape = tuple(X.shape[1:])
            break

        self.generator = Generator(self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        
    
    def train(self, epochs, adversarial_loss, n_threads, sample_interval, lr_opt, betas_opt):

        # Optimizers
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr_opt, betas=betas_opt)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr_opt, betas=betas_opt)
        if self.cuda:
            adversarial_loss.cuda()

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        for epoch in range(epochs):
            d_loss_cum=0
            g_loss_cum=0
            for i, (imgs, _) in enumerate(self.dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_generator.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_generator.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_discriminator.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_discriminator.step()

                

                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                
                d_loss_cum += d_loss.item()
                g_loss_cum += g_loss.item()

            d_loss_cum /= len(self.dataloader)
            g_loss_cum /= len(self.dataloader)
            print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, d_loss.item(), g_loss.item())
                )



if __name__=="__main__":

    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    opt = parser.parse_args()
    print(opt)

   
    cuda = torch.cuda.is_available()

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )


    

    gan = GAN(dataloader, opt.latent_dim)
    gan.train(opt.n_epochs, adversarial_loss, opt.n_cpu, opt.sample_interval, opt.lr, (opt.b1, opt.b2))
