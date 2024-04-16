import torch
import torch.nn.functional as F


from dataclasses import dataclass

def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.detach().cpu().numpy()


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func, rssm_state_mode):
        RSSMState = get_RSSMState(rssm_state_mode)
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return torch.distributions.independent.Independent(torch.distributions.Normal(self.mean, self.std), 1)


def get_RSSMState(rssm_state_mode):
    RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[rssm_state_mode]
    return RSSMState
