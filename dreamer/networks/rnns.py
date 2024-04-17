import torch
import torch.nn as nn

from dreamer.networks.layers import AttentionEncoder, DiscreteLatentDist
from utils.net_setup import get_active_func
from utils.trans_setup import get_RSSMState

class RSSMTransition(nn.Module):
    def __init__(self, args, action_size, hidden_size=200):
        super().__init__()
        """
        _rnn_input_model: (z_, a_) -> stoch_input
        _attention_stack: stoch_input -> attn
        _cell: (attn, h_) -> h
        _stochastic_prior_model: h -> (logits, z')
        return:  (logits, z', h)
        
        """
        self.args = args
        self._stoch_size = args["stochastic_size"]
        self._deter_size = args["deterministic_size"]
        self._hidden_size = hidden_size
        self._activaion = args['activation_func']
        self._attention_layers = args["attention_layers"]
        self._n_categoricals = args["n_categoricals"]
        self._n_classes = args["n_classes"]
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self._attention_stack = AttentionEncoder(self._attention_layers, hidden_size, hidden_size, dropout=0.1)
        self._rnn_input_model = self._build_rnn_input_model(action_size + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, self._n_categoricals, self._n_classes,
                                                          self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [get_active_func(self._activaion)]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size = prev_actions.shape[0]
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        if len(stoch_input.shape ) < 3:
            stoch_input = stoch_input.unsqueeze(-2)
        attn = self._attention_stack(stoch_input, mask=mask)

        deter_state = self._cell(attn.reshape(1, batch_size , -1),
                                 prev_states.deter.reshape(1, batch_size, -1))[0].reshape(batch_size, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state)
        return get_RSSMState(self.args["action_type"])(logits=logits, stoch=stoch_state, deter=deter_state)

class RSSMRepresentation(nn.Module):
    def __init__(self, args, transition_model: RSSMTransition):
        super().__init__()
        """
        _transition_model: (a_, z_) -> (logits, z', h)
        _stochastic_posterior_model: (h, embed(o)) -> logits, z
        posterior_states: (logits, z, h)
        return (z', z)
        """
        self.args = args
        self._transition_model = transition_model
        self._stoch_size = args["stochastic_size"]
        self._deter_size = args["deterministic_size"]
        self._embed_size = args["embed_size"]
        self._hidden_size = args["hidden_size"]
        self._n_categoricals = args["n_categoricals"]
        self._n_classes = args["n_classes"]
        self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + self._embed_size, self._n_categoricals, self._n_classes,
                                                              self._hidden_size)
        
    
    def forward(self, obs_embed, prev_actions, prev_states, mask=None):
        prior_states = self._transition_model(prev_actions, prev_states, mask)
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x)
        posterior_states = get_RSSMState(self.args["action_type"])(logits=logits, stoch=stoch_state, deter=prior_states.deter)
        return prior_states, posterior_states
    
    def initial_state(self, batch_size, **kwargs):
        RSSMState = get_RSSMState(self.args["action_type"])
        return RSSMState(stoch=torch.zeros(batch_size, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, self._deter_size, **kwargs))

    