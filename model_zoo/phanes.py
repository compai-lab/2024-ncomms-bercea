import torch
import torch.nn as nn
import torch.nn.functional as F

# standard
import matplotlib
matplotlib.use('Agg')
from model_zoo.aotgan import aotgan as net
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import GaussianBlur

from skimage import exposure
import copy
import lpips

""" Implementation of: Bercea, C. I., Neumayr, M., Rueckert, D. Schnabel, J. A. Mask, stitch, and re-sample:
Enhancing robustness and generalizability in anomaly detection through automatic diffusion models. ICML Workshops, 2023.
"""
class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = cond_dim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x, o_cond=None):
        y = x
        embeddings = []
        for layer in self.main:
            y = layer(y)
            if isinstance(layer, nn.AvgPool2d):
                embeddings.append(y)
        y = y.view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar, {'embeddings': embeddings}


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(zdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(zdim, num_fc_features),
                nn.ReLU(True),
            )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
        self.main.add_module('sigmoid', nn.Sigmoid())

    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class AnomalyMap:
    def __init__(self):
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).cuda()
        self.gfilter = GaussianBlur(3, sigma=(0.5, 3))
        super(AnomalyMap, self).__init__()

    def norm(self, input):
        input = input/np.max(input)
        return input

    def compute_residual(self, x_rec, x):
        saliency = self.get_saliency(x_rec, x)
        x_rescale = torch.Tensor(exposure.equalize_adapthist(x.cpu().detach().numpy())).to(x_rec.device)
        x_rec_rescale = torch.Tensor(exposure.equalize_adapthist(x_rec.cpu().detach().numpy())).to(x.device)
        x_res_2 = torch.abs(x_rec_rescale - x_rescale)
        x_res = x_res_2
        perc95 = torch.quantile(x_res, 0.95)
        eps=1e-8
        x_res = x_res / (perc95+eps)
        x_res[x_res > 1] = 1
        return x_res, saliency

    def get_saliency(self, x_rec, x):
        saliency = self.l_pips_sq(2*x_rec-1, 2*x-1)
        saliency = gaussian_filter(saliency.cpu().detach().numpy(), sigma=2)
        return torch.tensor(saliency).to(x.device)

    def filter_anomaly_mask(self, anomaly_masks):
        ### CAUTION: The masking threshold is essential for good performance, you can either set it to produce low
        # false positive, e.g., 95 percentile of the error on HEALTHY VALIDATION OR if no such analysis can be
        # conducted compute automatically the 95th percentile of each scan.

        masking_threshold = None # to be automatically determined  0.1 # training 0.153 # inference (set on the
        # validation set)

        # anomaly_masks[anomaly_masks>0.135] = 1  #for training # 0.1 for inference 0.153 (set on the validation set)
        # anomaly_masks[anomaly_masks<1] = 0
        # masking_threshold = 0.125
        filtered_masks = []
        for b in range(anomaly_masks.shape[0]):
            anomaly_mask = anomaly_masks[b][0].cpu().detach().numpy()
            masking_threshold = np.percentile(anomaly_mask, 95) if masking_threshold is None else masking_threshold
            anomaly_mask[anomaly_mask>masking_threshold] = 1
            anomaly_mask[anomaly_mask>1]=0
            filtered_mask = anomaly_mask
            filtered_masks.append(np.expand_dims(filtered_mask, 0))
        filtered_masks = gaussian_filter(np.asarray(filtered_masks), sigma=1.2)
        filtered_masks = torch.Tensor(filtered_masks).to(anomaly_masks.device)

        filtered_masks[filtered_masks>0.1]=1
        filtered_masks[filtered_masks<1]=0

        return filtered_masks


class Phanes(nn.Module):
    def __init__(self, cdim=1, zdim=128, channels=(64, 128, 256, 512, 512), image_size=128, conditional=False,
                 cond_dim=10):
        super(Phanes, self).__init__()

        self.zdim = zdim
        self.conditional = conditional
        self.cond_dim = cond_dim

        self.encoder = Encoder(cdim, zdim, channels, image_size, conditional=conditional, cond_dim=cond_dim)

        self.decoder = Decoder(cdim, zdim, channels, image_size, conditional=conditional,
                               conv_input_size=self.encoder.conv_output_size, cond_dim=cond_dim)

        self.ano_maps = AnomalyMap()

        in_channels: 1
        n_classes: 1
        norm: "group"
        up_mode: "upconv"
        depth: 4  # 4 for 128 x 128
        wf: 6  # 6 for 128x128
        padding: True
        self.netG = net.InpaintGenerator().cuda()
        self.netD = net.Discriminator().cuda()

    def forward(self, x, mask=None, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar, embed_dict = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar, embed_dict = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z).detach()
        x_res, saliency = self.ano_maps.compute_residual(y.detach(), x)

        x_res = (x_res) * saliency
        x_res_orig = copy.deepcopy(x_res.detach())
        ano_mask = self.ano_maps.filter_anomaly_mask(x_res)
        if mask is not None:
            ano_mask = mask
            print('Using masks...')

        transformed_images = copy.deepcopy(x.detach())
        transformed_images = (transformed_images * (1 - ano_mask).float()) + ano_mask

        y_hr = self.netG(transformed_images, ano_mask)
        y_hr = torch.clamp(y_hr, 0, 1)
        y_hr = (1 - ano_mask) * x + ano_mask * y_hr
        return y_hr, {'z_mu': mu, 'z_logvar': logvar,'z': z, 'embeddings': embed_dict['embeddings'], 'y_coarse': y,
                      'residual': x_res_orig, 'masked': transformed_images, 'saliency': saliency}

    def get_anomaly(self, x, mask=None, o_cond=None, deterministic=False):
        x_rec, x_dict = self.forward(x, mask, o_cond, deterministic)
        mask_ = x_dict['masked']
        mask_[mask_ < 1] = 0
        x_rec_ = x_rec.cpu().detach().numpy()
        x_ = x.cpu().detach().numpy()
        anomaly_maps = np.abs(x_ - x_rec_)
        # anomaly_maps *= x_dict['residual'].cpu().detach().numpy()
        anomaly_maps *= (self.ano_maps.get_saliency(x_rec, x).cpu().detach().numpy())
        anomaly_score = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        # anomaly_maps, anomaly_score = self.compute_anomaly(inputs, x_rec)
        return anomaly_maps, anomaly_score, {'x_rec': x_rec, 'mask': mask_, 'x_res': x_dict['residual'],
                                            'x_rec_orig': x_dict['y_coarse']}

    def ae(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar, embed_dict = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar, embed_dict = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z).detach()
        return y, {'z_mu': mu, 'z_logvar': logvar,'z': z, 'embeddings': embed_dict['embeddings']}

    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        if self.conditional and o_cond is not None:
            mu, logvar, embed_dict = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar, embed_dict = self.encoder(x)
        return mu, logvar, embed_dict

    def decode(self, z, y_cond=None):
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y


"""
Helpers
"""
def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error