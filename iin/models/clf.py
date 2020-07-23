import torch
import torch.nn as nn
import numpy as np
from edflow.util import retrieve

from iin.models.ae import FeatureLayer, DenseEncoderLayer, weights_init


class Distribution(object):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        n_down = retrieve(config, "Model/n_down")
        z_dim = retrieve(config, "Model/z_dim")
        in_size = retrieve(config, "Model/in_size")
        z_dim = retrieve(config, "Model/z_dim")
        bottleneck_size = in_size // 2**n_down
        in_channels = retrieve(config, "Model/in_channels")
        norm = retrieve(config, "Model/norm")
        n_classes = retrieve(config, "n_classes")

        self.feature_layers = nn.ModuleList()

        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))

        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, z_dim)
        self.classifier = torch.nn.Linear(z_dim, n_classes)

        self.apply(weights_init)

        self.n_down = n_down
        self.bottleneck_size = bottleneck_size

    def forward(self, input):
        h = self.encode(input).mode()
        assert h.shape[2] == h.shape[3] == 1
        h = h[:,:,0,0]
        h = self.classifier(h)
        return h
    
    def encode(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return Distribution(h)
