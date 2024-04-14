
import os
import numpy as np
import setproctitle

from ossmc.agent import Agent
from utils.configs_setup import get_task_name, init_dir, save_config
from utils.env_setup import make_train_env, get_num_agents, set_seed
from common.off_policy_buffer_fp import OffPolicyBufferFP
from dreamer.models import DreamModel
from utils.trans_setup import _t2n

class Runner:
    def __init__(self, args, marl_args, env_args, world_model_args) -> None:
        # Initialize config settings and path et.al.
        self.args = args
        self.marl_args = marl_args
        self.env_args = env_args
        self.world_model_args = world_model_args
        self.task_name = get_task_name(self.args['env'], self.env_args)
        self.run_dir, self.log_dir, self.save_dir, self.writter \
        = init_dir()
        save_config(self.args, self.marl_args, self.world_model_args, self.env_args, self.run_dir)
        self.log_file = open(
             os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        setproctitle.setproctitle(
            str(args["marl_algo"]) + "-" + str(args["world_model"]) + "-" + str(args["env"]) 
        )


        # Attribute regard of agent, env.
        self.envs = make_train_env()
        self.eval_envs = make_train_env()
        self.num_agents = get_num_agents(args["env"], env_args)
        self.agent_deaths = np.zeros(
            (self.marl_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )

        # Instance models.

        self.agent = Agent()
        self.buffer = OffPolicyBufferFP()
        self.world_model = DreamModel()





    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
            Args:
                available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                    shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            Returns:
                actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        


    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        obs, share_obs, available_actions = self.envs.reset()



    def get_actions(self):
        pass
    
    def insert(self):
        pass

    def run(self):
        set_seed(self.marl_args["seed"])
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        print("start warmup")

        obs, share_obs, available_actions = self.warmup()

        print(" start training")

        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )

        for step in range(1, steps + 1):
            actions = self.get_actions()

            (
            new_obs, new_share_obs, rewards, dones, infos, new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2),
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2),
                _t2n(self.messages)
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions

            if step % self.marl_args["train"]["train_interval"] == 0:
                for _ in range(self.world_model_args.model_epochs):
                    self.train_world_model()
                for _ in range(update_num):
                    self.train_agent()


