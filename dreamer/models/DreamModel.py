import torch
import torch.nn as nn

from dreamer.networks.dense import DenseBinaryModel, DenseModel
from dreamer.networks.vae import Encoder, Decoder
from dreamer.networks.rnns import RSSMRepresentation, RSSMTransition
from utils.env_setup import get_shape_from_obs_space

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
        self.to(device)

    
    def forward(self, observations, prev_actions=None, prev_states=None, mask=None):
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), observations.size(1), self.action_size,
                                       device=observations.device)

        if prev_states is None:
            prev_states = self.representation.initial_state(prev_actions.size(0), observations.size(1),
                                                            device=observations.device)
        obs_embeds = self.observation_encoder(observations)
        _, states = self.representation(obs_embeds, prev_actions, prev_states, mask)
        
