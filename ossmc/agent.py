
import torch
import numpy as np

from utils.env_setup import init_device, get_num_agents
from common.valuenorm import ValueNorm
from common.off_policy_buffer_fp import OffPolicyBufferFP
from ossmc.algorithms.ossac import OSSAC as Policy
from ossmc.algorithms.critic import SoftTwinContinuousQCritic as Critic
from dreamer.models.DreamModel import DreamerModel

class Agent:
    def __init__(self, envs, args, marl_args, env_args, wm_args) -> None:
        self.args = args
        self.env_args = env_args
        self.marl_args = marl_args
        self.envs = envs
        self.wm_args = wm_args

        if "policy_freq" in self.marl_args["algo"]:
            self.policy_freq = self.marl_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = self.env_args.get("state_type", "EP")
        self.share_param = self.marl_args["algo"]["share_param"]
        self.fixed_order = self.marl_args["algo"]["fixed_order"]

        
        self.device = init_device(self.marl_args["device"])

        if (
                "use_valuenorm" in self.marl_args["train"].keys()
                and self.marl_args["train"]["use_valuenorm"]
            ):
                self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.num_agents = get_num_agents(args["env"], env_args)
        
        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(self.marl_args["seed"]["seed"] + agent_id + 1)


        self.actor = []
        self.world_model = []
        for agent_id in range(self.num_agents):
            # individual policy construction: feat -> a, log_prob(a)
            agent = Policy(
                {**marl_args["model"], **marl_args["algo"]},
                self.wm_args["feat_size"],
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.actor.append(agent)
            # individual dreamer model construction: obs, s_ -> (z, z')
            wm = DreamerModel(
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
                self.args,
                self.env_args,
                self.wm_args,
                device=self.device,
                )
            self.world_model.append(wm)

        # global critic construction: feat -> Q(feat, a)
        self.critic = Critic(
             {**marl_args["train"], **marl_args["model"], **marl_args["algo"]},
                [self.wm_args["feat_size"] * self.num_agents],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
        )


        self.total_it = 0
        if (
            "auto_alpha" in self.marl_args["algo"].keys()
            and self.marl_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                 self.target_entropy.append(
                        -0.98
                        * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
                    )  

            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.num_agents):
                _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha], lr=self.marl_args["algo"]["alpha_lr"]
                    )
                )
                self.alpha.append(torch.exp(_log_alpha.detach()))

        elif "alpha" in self.marl_args["algo"].keys():
            self.alpha = [self.marl_args["algo"]["alpha"]] * self.num_agents


    def train_ac(self, data):
        self.total_it += 1
        (
            sp_share_obs, sp_obs, sp_actions, sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward, sp_done, sp_valid_transition, sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs, sp_next_obs, sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,
        ) = data

        self.critic.turn_on_grad()
        all_actions = []
        all_logp_actions = []
        all_feats = []
        all_av_actions = []
        all_reward = []
        all_discount = []
        for agent_id in range(self.num_agents):
            (next_states, actions, av_actions, pis) = self.world_model[agent_id].generate_with_policy(
                (sp_obs[agent_id], sp_actions[agent_id], sp_done[agent_id]), self.actor[agent_id]
                )
            all_actions.append(actions)
            feat = next_states.get_features()
            reward = self.world_model[agent_id].reward_model(feat)
            discount = self.world_model[agent_id].pcont(feat).mean

            all_feats.append(feat)
            all_logp_actions.append(pis)
            all_av_actions.append(av_actions)
            all_reward.append(reward)
            all_discount.append(discount)
               
        imag_feats = torch.cat(all_feats, dim=-1)
        imag_reward = torch.stack(all_reward, dim=0).mean(dim=0)[:-1]
        imag_discount = torch.stack(all_discount, dim=0).mean(dim=0)[1:]

        cur_feats = imag_feats[:-1]
        cur_actions = [a[:-1] for a in all_actions]

        next_feats = imag_feats[1:]
        next_actions = [a[1:] for a in all_actions]
        next_logp_actions = [a[1:] for a in all_logp_actions]

        
        self.critic.train(
            cur_feats,
            cur_actions,
            imag_reward,
            imag_discount,
            next_feats,
            next_actions,
            next_logp_actions,
        )
        self.critic.turn_off_grad()

        if self.total_it % self.policy_freq == 0:
            actions = []
            logp_actions = []
            feats = []

            
            for agent_id in range(self.num_agents): 
                (next_state, action, _, logp) = self.world_model[agent_id].generate_with_policy(
                    (sp_obs[agent_id], sp_actions[agent_id], sp_done[agent_id]), self.actor[agent_id]
                    )
                actions.append(action)
                logp_actions.append(logp)
                feat = next_state.get_features()
                feats.append(feat)
                
            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(np.random.permutation(self.num_agents))
            for agent_id in agent_order:
                self.actor[agent_id].turn_on_grad()
                (next_state, action, _, logp) = self.world_model[agent_id].generate_with_policy(
                    (sp_obs[agent_id], sp_actions[agent_id], sp_done[agent_id]), self.actor[agent_id]
                    )
                actions[agent_id] = action
                logp_actions[agent_id] = logp
                feat = next_state.get_features()
                feats[agent_id] = feat
            
                value_pred = self.critic.get_values(feats, actions)

                actor_loss = -torch.mean(
                    value_pred - self.alpha[agent_id] * logp_actions[agent_id]
                )
                
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()

                # train this agent's alpha
                if self.marl_args["algo"]["auto_alpha"]:
                    log_prob = (
                        logp_actions[agent_id].detach()
                        + self.target_entropy[agent_id]
                    )
                    alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                    self.alpha_optimizer[agent_id].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[agent_id].step()
                    self.alpha[agent_id] = torch.exp(
                        self.log_alpha[agent_id].detach()
                    )

                actions[agent_id] = action.detach()

            # train critic's alpha
            if self.marl_args["algo"]["auto_alpha"]:
                self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            self.critic.soft_update()    


        
