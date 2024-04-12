import torch
import torch.nn as nn

from dreamer.networks.dense import DenseBinaryModel, DenseModel
from dreamer.networks.vae import Encoder, Decoder
from dreamer.networks.rnns import RSSMRepresentation, RSSMTransition


class DreamerModel(nn.Module):
    def __init__(self, config):
        super().__init__()