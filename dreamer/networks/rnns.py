import torch
import torch.nn as nn

from dreamer.networks.layers import AttentionEncoder, DiscreteLatentDist
from utils.net_setup import get_active_func

class RSSMTransition(nn.Module):
    def __init__(self, args, action_size, hidden_size=200):
        super().__init__()
        self._stoch_size = args.stochastic_size
        self._deter_size = args.deterministic_size
        self._hidden_size = hidden_size
        self._activaion = args['activation_func']
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self._attention_stack = AttentionEncoder(args.attention_layers, hidden_size, hidden_size, dropout=0.1)
        self._rnn_input_model = self._build_rnn_input_model(action_size + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, args.n_categoricals, args.n_classes,
                                                          self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [get_active_func(self._activaion)]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size = prev_actions.shape[0]
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        attn = self._attention_stack(stoch_input, mask=mask)
        deter_state = self._cell(attn.reshape(1, batch_size , -1),
                                 prev_states.deter.reshape(1, batch_size, -1))[0].reshape(batch_size, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state)


class RSSMRepresentation(nn.Module):
    def __init__(self, args, transition_model: RSSMTransition):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = args.stochastic_size
        self._deter_size = args.deterministic_size
        self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + args.embed_size, args.n_categoricals, args.n_classes,
                                                              args.hidden_size)
        
    
    def forward(self, obs_embed, prev_actions, prev_states, mask=None):
        prior_states = self._transition_model(prev_actions, prev_states, mask)
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x)