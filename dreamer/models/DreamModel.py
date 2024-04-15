import torch
import torch.nn as nn

from dreamer.networks.dense import DenseBinaryModel, DenseModel
from dreamer.networks.vae import Encoder, Decoder
from dreamer.networks.rnns import RSSMRepresentation, RSSMTransition


class DreamerModel(nn.Module):
    def __init__(self, obs_shape, action_space, args, env_args, wm_args):
        super().__init__() 
        self.args = args
        self.env_args = env_args
        self.wm_args = wm_args

        self.action_size = action_space

        self.obs_encoder = Encoder(obs_shape, self.wm_args.hidden_size
                                   , self.wm_args.embed_size)
        self.obs_decoder = Decoder(self.wm_args.feat_size, self.wm_args.hidden_size
                                   , obs_shape)
        self.transition = RSSMTransition(self.wm_args)
        self.representation = RSSMRepresentation(self.wm_args, self.transition)
        self.reward_model = DenseModel(wm_args.feat_size, 1, wm_args.reward_layers, wm_args.reward_hidden)
        self.pcont = DenseBinaryModel(wm_args.feat_size, 1, wm_args.pcont_layers, wm_args.pcont_hidden)

        if  "smac" in args["env_name"]:
            self.av_action = DenseBinaryModel(wm_args.feat_size, self.action_size, wm_args.pcont_layers, wm_args.pcont_hidden)  
        else:
            self.av_action = None
        self.q_features = DenseModel(wm_args.hidden, wm_args.pcont_hidden, 1, wm_args.pcont_hidden)
        self.q_action = nn.Linear(wm_args.pcont_hidden, self.action_size)


    
    def forward(self, observations, prev_actions=None, prev_states=None, mask=None):
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), observations.size(1), self.action_size,
                                       device=observations.device)

        if prev_states is None:
            prev_states = self.representation.initial_state(prev_actions.size(0), observations.size(1),
                                                            device=observations.device)
        obs_embeds = self.observation_encoder(observations)
        _, states = self.representation(obs_embeds, prev_actions, prev_states, mask)
        
