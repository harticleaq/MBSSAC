import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils.env_setup import check, get_shape_from_obs_space
from utils.discrete_setup import gumbel_softmax
from ossmc.algorithms.networks import MLPBase, ACTLayer

class Actor(nn.Module):
    def __init__(self, args, feat_size, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_dim = feat_size
        self.base = MLPBase(args, obs_dim)
        act_dim = self.hidden_sizes[-1]
        self.act = ACTLayer(
            action_space,
            act_dim,
            self.initialization_method,
            self.gain,
        )
        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True, ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

   
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic,  
        )

        return actions
    
    def get_logits(self, obs, available_actions=None):
        """Get action logits from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) input to network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        actor_features = self.base(obs)

        return self.act.get_logits(actor_features, available_actions)




class OSSAC:
    def __init__(self, args, feat_size, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        self.actor = Actor(args, feat_size, act_space, device)
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)

        actions = self.actor(obs, available_actions, 
              stochastic)
        return actions
    
    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        logits = self.actor.get_logits(obs, available_actions)
        actions = gumbel_softmax(
            logits, hard=True, device=self.device
        )  # onehot actions
        logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        return actions, logp_actions
    


    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True
 

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False


    def soft_update(self):
        """Soft update target actor."""
        for param_target, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "\\actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "\\actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
        