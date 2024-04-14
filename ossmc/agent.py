from utils.env_setup import init_device, get_num_agents
from common.valuenorm import ValueNorm

class Agent:
    def __init__(self, envs, args, marl_args, env_args) -> None:
        self.args = args
        self.env_args = env_args
        self.marl_args = marl_args
        self.envs = envs

        if "policy_freq" in self.marl_args["algo"]:
            self.policy_freq = self.marl_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = self.env_args.get("state_type", "EP")
        self.share_param = self.marl_args["algo"]["share_param"]
        self.fixed_order = self.marl_args["algo"]["fixed_order"]

        
        self.device = init_device(self.marl_args["device"])

        if (
                "use_valuenorm" in self.algo_args["train"].keys()
                and self.algo_args["train"]["use_valuenorm"]
            ):
                self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        
        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(self.marl_args["seed"]["seed"] + agent_id + 1)


        self.actor = []
        


        