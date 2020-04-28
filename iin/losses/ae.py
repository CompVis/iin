import numpy as np
from edflow.util import retrieve
import torch.nn as nn
import torch
from perceptual_similarity import PerceptualLoss
import functools
from torch.nn.utils.spectral_norm import spectral_norm


def do_spectral_norm(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        spectral_norm(m)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def l1(input, target):
    return torch.abs(input-target)


def l2(input, target):
    return torch.pow((input-target), 2)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
        --> from
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LossPND(nn.Module):
    """Using LPIPS Perceptual loss"""
    def __init__(self, config):
        super().__init__()
        __pixel_loss_opt = {"l1": l1,
                            "l2": l2 }
        self.config = config
        self.discriminator_iter_start = retrieve(config, "Loss/disc_start")
        self.disc_factor = retrieve(config, "Loss/disc_factor", default=1.0)
        self.kl_weight = retrieve(config, "Loss/kl_weight", default=1.0)
        self.perceptual_weight = retrieve(config, "Loss/perceptual_weight", default=1.0)
        if self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        self.calibrate = retrieve(config, "Loss/calibrate", default=False)
        # output log variance
        self.logvar_init = retrieve(config, "Loss/logvar_init", default=0.0)
        self.logvar = nn.Parameter(torch.ones(size=())*self.logvar_init)
        # discriminator
        self.discriminator_weight = retrieve(config, "Loss/discriminator_weight", default=1.0)
        disc_nc_in = retrieve(config, "Loss/disc_in_channels", default=3)
        disc_layers = retrieve(config, "Loss/disc_num_layers", default=3)
        self.discriminator = NLayerDiscriminator(input_nc=disc_nc_in, n_layers=disc_layers).apply(weights_init)
        self.pixel_loss = __pixel_loss_opt[retrieve(config, "Loss/pixelloss", default="l1")]
        if retrieve(config, "Loss/spectral_norm", default=False):
            self.discriminator.apply(do_spectral_norm)
        if torch.cuda.is_available():
            self.discriminator.cuda()
        if "learning_rate" in self.config:
            learning_rate = self.config["learning_rate"]
        elif "base_learning_rate" in self.config:
            learning_rate = self.config["base_learning_rate"]*self.config["batch_size"]
        else:
            learning_rate = 0.001
        self.learning_rate = retrieve(config, "Loss/d_lr_factor", default=1.0)*learning_rate
        self.num_steps = retrieve(self.config, "num_steps")
        self.decay_start = retrieve(self.config, "decay_start", default=self.num_steps)
        self.d_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.9))

    def get_decay_factor(self, global_step):
        alpha = 1.0
        if self.num_steps > self.decay_start:
            alpha = 1.0 - np.clip(
                (global_step - self.decay_start) /
                (self.num_steps - self.decay_start),
                0.0, 1.0)
        return alpha

    def update_lr(self, global_step):
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = self.get_decay_factor(global_step)*self.learning_rate

    def set_last_layer(self, last_layer):
        self.last_layer = [last_layer]

    def parameters(self):
        """Exclude discriminator from parameters."""
        ps = super().parameters()
        exclude = set(self.discriminator.parameters())
        ps = (p for p in ps if not p in exclude)
        return ps

    def forward(self, inputs, reconstructions, posteriors, global_step):
        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        rec_loss = self.pixel_loss(inputs, reconstructions)   # l1 or l2
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight*p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        calibration = rec_loss / torch.exp(self.logvar) - 1.0
        calibration = torch.sum(calibration) / calibration.shape[0]

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        logits_real = self.discriminator(inputs)
        logits_fake = self.discriminator(reconstructions)
        d_loss = 0.5*(
                torch.mean(torch.nn.functional.softplus(-logits_real)) +
                torch.mean(torch.nn.functional.softplus( logits_fake))) * disc_factor
        def train_op():
            self.d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
            self.update_lr(global_step)

        g_loss = -torch.mean(logits_fake)

        if not self.calibrate:
            loss = nll_loss + self.kl_weight*kl_loss + self.discriminator_weight*g_loss*disc_factor
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
            d_weight = torch.norm(nll_grads)/(torch.norm(g_grads)+1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            loss = nll_loss + self.kl_weight*kl_loss + d_weight*disc_factor*g_loss
        log = {"scalars": {"loss": loss, "logvar": self.logvar,
                           "kl_loss": kl_loss, "nll_loss": nll_loss,
                           "rec_loss": rec_loss.mean(),
                           "calibration": calibration,
                           "g_loss": g_loss, "d_loss": d_loss,
                           "logits_real": torch.mean(logits_real),
                           "logits_fake": torch.mean(logits_fake),
                           }}
        if self.calibrate:
            log["scalars"]["d_weight"] = d_weight
        for i, param_group in enumerate(self.d_optimizer.param_groups):
            log["scalars"]["d_lr_{:02}".format(i)] = param_group['lr']
        return loss, log, train_op
