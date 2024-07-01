import torch.nn as nn
import numpy as np
import torch


"""
The code is based on:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


class WGAN(nn.Module):
    def __init__(self, image_size=128, latent_dim=100, channels=1):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        opt = {'img_size': image_size, 'latent_dim': latent_dim, 'channels': channels}
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)

    def forward(self, x):
        return self.generator(x)


class fAnoGAN(nn.Module):
    def __init__(self, image_size, latent_dim, channels=1, wgan_path=''):
        super(fAnoGAN, self).__init__()
        opt = {'img_size': image_size, 'latent_dim': latent_dim, 'channels': channels}
        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(opt)
        self.wgan = WGAN(image_size, latent_dim, channels)
        self.wgan.to(self.device)
        wgan_checkpoint = torch.load(wgan_path, map_location=torch.device(self.device))['model_weights']
        self.wgan.load_state_dict(wgan_checkpoint)
        self.wgan.eval()
        self.criterion = nn.MSELoss()
        self.kappa = 1.0

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.wgan.generator(z)
        return x_rec, {'z': z}

    def get_anomaly(self, x):
        real_z = self.encoder(x)
        x_rec = self.wgan.generator(real_z)
        x_rec_ = x_rec.cpu().detach().numpy()
        x_ = x.cpu().detach().numpy()
        anomaly_maps = np.abs(x_ - x_rec_)
        anomaly_score = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        # real_feature = self.wgan.discriminator.forward_features(x)
        # fake_feature = self.wgan.discriminator.forward_features(x_rec)
        # Scores for anomaly detection
        # img_distance = self.criterion(x_rec, x)
        # loss_feature = self.criterion(fake_feature, real_feature)
        # anomaly_maps = torch.abs(x-x_rec)
        # anomaly_score = img_distance + self.kappa * loss_feature
        # anomaly_score = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        return anomaly_maps, anomaly_score, {'x_rec': x_rec}


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.init_size = opt['img_size'] // 4
        self.l1 = nn.Sequential(nn.Linear(opt['latent_dim'],
                                128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt['channels'], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        # print(out.shape)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt['channels'], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt['img_size'] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        return features


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(opt['channels'], 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt['img_size'] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2,
                                                 opt['latent_dim']),
                                       nn.Tanh())

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity