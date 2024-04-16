
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
            agent = Policy(
                {**marl_args["model"], **marl_args["algo"]},
                self.wm_args["feat_size"],
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.actor.append(agent)
            wm = DreamerModel(
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
                self.args,
                self.env_args,
                self.wm_args,
                device=self.device,
                )
            self.world_model.append(wm)
    
        self.critic = Critic(
             {**marl_args["train"], **marl_args["model"], **marl_args["algo"]},
                self.envs.share_observation_space[0],
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


    def train(self):
        self.total_it += 1