"""ValueNorm."""
import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""
    def __init__(self, input_shape, norm_axes=1, beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"), 
        ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )