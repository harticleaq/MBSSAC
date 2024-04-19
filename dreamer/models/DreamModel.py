import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dreamer.networks.dense import DenseBinaryModel, DenseModel
from dreamer.networks.vae import Encoder, Decoder
from dreamer.networks.rnns import RSSMRepresentation, RSSMTransition
from utils.env_setup import get_shape_from_obs_space, check
from utils.trans_setup import stack_states

class DreamerModel(nn.Module):
    def __init__(self, obs_space, action_space, args, env_args, wm_args, device):
        super().__init__() 
        self.args = args
        self.env_args = env_args
        self.wm_args = wm_args

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_shape = obs_shape[0] 
        self.action_type = action_space.__class__.__name__
        self.action_size = action_space.n

        self.hidden_size = self.wm_args['hidden_size']
        self.embed_size = self.wm_args["embed_size"]
        self.feat_size = wm_args["feat_size"]
        self.reward_layers = wm_args["reward_layers"]
        self.reward_hidden = wm_args["reward_hidden"]
        self.pcont_layers = wm_args["pcont_layers"]
        self.pcont_hidden = wm_args["pcont_hidden"]

        self.obs_encoder = Encoder(obs_shape, self.hidden_size, self.embed_size)  
        self.obs_decoder = Decoder(self.feat_size, self.hidden_size, obs_shape)
        self.transition = RSSMTransition(self.wm_args, self.action_size, self.hidden_size)
        self.representation = RSSMRepresentation(self.wm_args, self.transition)
        self.reward_model = DenseModel(self.feat_size, 1, self.reward_layers, self.reward_hidden)
        self.pcont = DenseBinaryModel(self.feat_size, 1, self.pcont_layers, self.pcont_hidden)

        if  "smac" in args["env"]:
            self.av_action = DenseBinaryModel(self.feat_size, self.action_size, self.pcont_layers, self.pcont_hidden)  
        else:
            self.av_action = None
        self.q_features = DenseModel(self.hidden_size, self.pcont_hidden, 1, self.pcont_hidden)
        self.q_action = nn.Linear(self.pcont_hidden, self.action_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.wm_args["lr"])
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.turn_off_grad()
        self.to(device)

    
    def forward(self, observations, prev_actions=None, prev_states=None, mask=None):
        observations = check(observations).to(**self.tpdv)
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), self.action_size,
                                       device=observations.device)

        if prev_states is None:
            prev_states = self.representation.initial_state(prev_actions.size(0),
                                                            device=observations.device)
        obs_embeds = self.obs_encoder(observations)
        _, states = self.representation(obs_embeds, prev_actions, prev_states, mask)
        return states

    def rollout_representation(self, 
            representation_model, steps, obs_embed, action, prev_states, done
        ):
        priors = []
        posteriors = []
        for t in range(steps):
            prior_states, posterior_states = representation_model(obs_embed[t], action[t], prev_states)
            prev_states = posterior_states.map(lambda x: x * (1.0 - done[t]))
            priors.append(prior_states)
            posteriors.append(posterior_states)
        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior.map(lambda x: x[:]), post.map(lambda x: x[:]), post.deter[:]

    def kl_div_categorical(self, p, q):
        eps = 1e-7
        return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


    def reshape_dist(self, dist):
        return dist.get_dist(dist.stoch.shape[:-1], self.wm_args["n_categoricals"], self.wm_args["n_classes"])

    def generate_with_policy(self, data, policy):
        (sp_obs, sp_actions, sp_done) = data
        time_steps = sp_obs.shape[0]
        batch_size = sp_obs.shape[1]
        sp_obs = sp_obs.reshape(time_steps, batch_size, -1)
        sp_actions = sp_actions.reshape(time_steps, batch_size, -1)
        sp_done = sp_done.reshape(time_steps, batch_size, -1)

        sp_obs = check(sp_obs).to(**self.tpdv)
        sp_actions = check(sp_actions).to(**self.tpdv_a)
        onehot_actions = F.one_hot(
                sp_actions, num_classes=self.action_size
                ).squeeze(-2)
        sp_done = check(sp_done).to(**self.tpdv)

        embed = self.obs_encoder(sp_obs.reshape(-1, sp_obs.shape[-1]))
        embed = embed.reshape(time_steps, batch_size, -1)
        prev_state = self.representation.initial_state(batch_size, device=sp_obs.device) # h
        _, post, _ = self.rollout_representation(self.representation, time_steps, embed, onehot_actions, prev_state, sp_done)
        post = post.map(lambda x: x.reshape(time_steps * batch_size, -1))
        state = post
        next_states = []
        actions = []
        av_actions = []
        pis = []
        for t in range(self.wm_args["horizon"]):
            feat = state.get_features()
            if self.av_action is not None:
                avail_actions = self.av_action(feat).sample()
                av_actions.append(avail_actions.squeeze(0))
            else:
                avail_actions = None
            action, pi = policy.get_actions_with_logprobs(feat, avail_actions)
            next_states.append(state)
            pis.append(pi)
            actions.append(action)
            state = self.transition(action.detach(), state)
        return (stack_states(next_states, dim=0), torch.stack(actions, dim=0),torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None
                ,torch.stack(pis, dim=0))


    def train_model(self, data):
        self.turn_on_grad()
        (
            sp_obs, sp_actions, sp_available_actions, sp_reward, sp_done
        ) = data

        time_steps = sp_obs.shape[0]
        batch_size = sp_obs.shape[1]
        sp_obs = sp_obs.reshape(time_steps, batch_size, -1)
        sp_actions = sp_actions.reshape(time_steps, batch_size, -1)
        sp_done = sp_done.reshape(time_steps, batch_size, -1)
        sp_reward = sp_reward.reshape(time_steps, batch_size, -1)
        sp_available_actions = sp_available_actions.reshape(time_steps, batch_size, -1)


        sp_obs = check(sp_obs).to(**self.tpdv)
        sp_actions = check(sp_actions).to(**self.tpdv_a)
        onehot_actions = F.one_hot(
                sp_actions, num_classes=self.action_size
                ).squeeze(-2)
        sp_done = check(sp_done).to(**self.tpdv)
        sp_reward = check(sp_reward).to(**self.tpdv)
        sp_available_actions = check(sp_available_actions).to(**self.tpdv)


        embed = self.obs_encoder(sp_obs.reshape(-1, sp_obs.shape[-1]))
        embed = embed.reshape(time_steps, batch_size, -1)
        prev_state = self.representation.initial_state(batch_size, device=sp_obs.device) # h
        prior, post, deters = self.rollout_representation(self.representation, time_steps, embed, onehot_actions, prev_state, sp_done)
    
        feat = torch.cat([post.stoch, deters], -1) # (z, h)
        feat_dec = post.get_features() # (z, h)

        # 重构损失 (feat_dec -> x', i_feat)
        x_pred, i_feat = self.obs_decoder(feat_dec.reshape(-1, feat_dec.shape[-1]))
        x = sp_obs.reshape(-1, sp_obs.shape[-1])
        rec_loss = (F.smooth_l1_loss(x_pred, x, reduction='none') * (1. - sp_done.reshape(-1, 1))).sum() / np.prod(list(x.shape[:-1]))
        
        
        # 奖励损失 (feat -> r')
        reward_loss = F.smooth_l1_loss(self.reward_model(feat), sp_reward)

        # 折扣损失 (feat -> gamma')
        pred_pcont = self.pcont(feat)
        pcont_loss = -torch.mean(pred_pcont.log_prob(1. - sp_done))

        # 可执行动作损失 (feat_dec -> a')
        pred_av_action = self.av_action(feat_dec)
        av_action_loss = -torch.mean(pred_av_action.log_prob(sp_available_actions)) if sp_available_actions is not None else 0.

        # 动作损失 
        i_feat = i_feat.reshape(time_steps, batch_size, -1)
        q_feat = F.relu(self.q_features(i_feat))
        action_logits = self.q_action(q_feat)
        criterion = nn.CrossEntropyLoss(reduction='none')
        action_loss = (1. - sp_done.reshape(-1)) * criterion(action_logits.view(-1, action_logits.shape[-1]), sp_actions.view(-1)) 
        action_loss = torch.mean(action_loss)

        # 散度损失
        prior_dist = self.reshape_dist(prior)
        post_dist = self.reshape_dist(post)
        post = self.kl_div_categorical(post_dist, prior_dist.detach())
        pri = self.kl_div_categorical(post_dist.detach(), prior_dist)
        kl_div = self.wm_args["kl_balance"] * post.mean(-1) + (1 - self.wm_args["kl_balance"]) * pri.mean(-1)
        kl_div = torch.mean(kl_div)

        # overall loss
        loss = kl_div + reward_loss + action_loss + rec_loss + pcont_loss + av_action_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.wm_args["grad_clip"])
        self.optimizer.step()
        self.turn_off_grad()
    
    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.parameters():
            param.requires_grad = False

