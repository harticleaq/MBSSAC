import torch.nn as nn
from utils.net_setup import get_init_method, get_active_func, init

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.args = args
        self.use_feature_normalization = self.args["use_feature_normalization"]
        self.initialization_method = self.args["initialization_method"]
        self.activation_func = self.args["activation_func"]
        self.hidden_sizes = self.args["hidden_sizes"]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape)

        self.mlp = MLPLayer(
            obs_shape, self.hidden_sizes, self.initialization_method,
            self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        
        x = self.mlp(x)
        return x

class ACTLayer(nn.Module):
    def __init__(self, 
        action_space, inputs_dim, initialization_method, gain    
                 ):
        super(ACTLayer).__init__()