import torch
import torch.nn as nn

from utils.net_setup import get_init_method, get_active_func, init
from utils.distributions import Categorical
from utils.env_setup import get_shape_from_obs_space, get_combined_dim

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
    def __init__(
            self, action_space, inputs_dim, initialization_method, gain    
        ):
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        action_dim = action_space.n

        self.action_out = Categorical(
            inputs_dim, action_dim, initialization_method, gain
        )

    def forward(self, x, available_actions=None, deterministic=False):
        action_distribution = self.action_out(x, available_actions)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)
        return actions, action_log_probs

    def get_logits(self, x, available_actions=None):

        action_distribution = self.action_out(x, available_actions)
        action_logits = action_distribution.logits
        return action_logits
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PlainCNN(nn.Module):
    def __init__(
        self, obs_shape, hidden_size, activation_func, kernel_size=3, stride=1
    ):
        super().__init__()
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        layers = [
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=hidden_size // 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            get_active_func(activation_func),
            Flatten(),
            nn.Linear(
                hidden_size
                // 4
                * (input_width - kernel_size + stride)
                * (input_height - kernel_size + stride),
                hidden_size,
            ),
            get_active_func(activation_func),
        ]
        self.cnn = nn.Sequential(*layers)
    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x

class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """
    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        self.args = args
        self.activation_func = self.args["activation_func"]
        self.hidden_sizes = self.args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                cent_obs_shape, self.hidden_sizes[0], self.activation_func
            )
            cent_obs_feature_dim = self.hidden_sizes[0]
        else:
            self.feature_extractor = None
            cent_obs_feature_dim = cent_obs_shape[0]
        
        sizes = (
            [get_combined_dim(cent_obs_feature_dim, act_spaces)]
            + list(self.hidden_sizes)
            + [1]
        )

        self.mlp = PlainMLP(sizes, self.activation_func)
        self.to(device)
    
    def forward(self, cent_obs, actions):
        if self.feature_extractor is not None:
            feature = self.feature_extractor(cent_obs)
        else:
            feature = cent_obs
        feature = torch.cat([feature, actions], dim=-1)
        q_values = self.mlp(feature)
        return q_values