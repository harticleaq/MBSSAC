import torch
import itertools

from copy import deepcopy
from algorithms.networks import ContinuousQNet

class TwinContinuousQCritic:
    """Twin Continuous Q Critic.
    Critic that learns two Q-functions. The action space is continuous.
    Note that the name TwinContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """
    def __init__(
            self,
            args,
            share_obs_space,
            act_space,
            num_agents,
            state_type,
            device=torch.device("cpu"),
        ):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = self.act_space[0].__class__.__name__
        self.critic = ContinuousQNet(args, share_obs_space, self.act_space, device)
        self.critic2 = ContinuousQNet(args, share_obs_space, self.act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()
    


    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False



class SoftTwinContinuousQCritic(TwinContinuousQCritic):
    """Soft Twin Continuous Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    Note that the name SoftTwinContinuousQCritic emphasizes its structure that takes observations and actions as input
    and outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be
    used in discrete action space.
    """

    def __init__(
        self, args, share_obs_space, act_space, num_agents, state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        super(SoftTwinContinuousQCritic, self).__init__(
            args, share_obs_space, act_space, num_agents, state_type, device
        )
        self.args = args
        self.tpdv = dict(dtype=torch.int64, device=device)
        self.auto_alpha = self.args["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args["alpha_lr"]
            )
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = args["alpha"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]
