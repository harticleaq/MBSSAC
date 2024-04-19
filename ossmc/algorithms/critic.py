import torch
import itertools
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from ossmc.algorithms.networks import ContinuousQNet
from utils.env_setup import check

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
    
    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

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
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
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
    
    def update_alpha(self, logp_actions, target_entropy):
        """Auto-tune the temperature parameter alpha."""
        log_prob = (
            torch.sum(torch.cat(logp_actions, dim=-1), dim=-1, keepdim=True)
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())
    
    def get_values(self, share_obs, actions):
        """Get the Q values for the given observations and actions."""
        imag_feat = torch.cat(share_obs, dim=-1)
        imag_action = torch.cat(actions, dim=-1)
        return torch.min(
            self.critic(imag_feat, imag_action), self.critic2(imag_feat, imag_action)
        )
    
    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma=0.99,
        value_normalizer=None,
    ):
        """Train the critic.
        Args:
            share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            actions: (n_agents, batch_size, dim)
            reward: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            done: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            valid_transition: (n_agents, batch_size, 1)
            term: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            next_share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            next_actions: (n_agents, batch_size, dim)
            next_logp_actions: (n_agents, batch_size, 1)
            gamma: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        """
        if self.action_type == "Box":
            actions = check(actions).to(**self.tpdv)
            actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        else:
            actions = torch.cat(actions, dim=-1) 

        if self.action_type == "Box":
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        else:
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv_a)
        next_logp_actions = torch.sum(
            torch.cat(next_logp_actions, dim=-1), dim=-1, keepdim=True
        ).to(**self.tpdv)


        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)

        if value_normalizer is not None:
            q_targets = reward + gamma * (
                check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                - self.alpha * next_logp_actions
            ) * (done)
            value_normalizer.update(q_targets)
            q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
        else:
            q_targets = reward + gamma * (
                next_q_values - self.alpha * next_logp_actions
            ) * (done)
        if self.use_huber_loss:
            if self.state_type == "FP" :
                critic_loss1 = (
                    torch.mean(
                        F.huber_loss(
                            self.critic(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                    )
                )
                critic_loss2 = (
                    torch.mean(
                        F.huber_loss(
                            self.critic2(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                    )
                )
            else:
                critic_loss1 = torch.mean(
                    F.huber_loss(
                        self.critic(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(
                        self.critic2(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if self.state_type == "FP" :
                critic_loss1 = (
                    torch.mean(
                        F.mse_loss(self.critic(share_obs, actions), q_targets)
                    )
                )
                critic_loss2 = (
                    torch.mean(
                        F.mse_loss(self.critic2(share_obs, actions), q_targets)
                    )
                )
            else:
                critic_loss1 = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets)
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()