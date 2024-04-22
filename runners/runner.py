
import os
import torch
import numpy as np
import setproctitle
import torch.nn.functional as F

from copy import deepcopy
from ossmc.agent import Agent
from utils.configs_setup import get_task_name, init_dir, save_config
from utils.env_setup import make_train_env, get_num_agents, set_seed
from common.off_policy_buffer_fp import OffPolicyBufferFP
from utils.trans_setup import _t2n
from torch.distributions import Categorical

class Runner:
    def __init__(self, args, marl_args, env_args, world_model_args):
        # Initialize config settings and path et.al.
        self.args = args
        self.marl_args = marl_args
        self.env_args = env_args
        self.world_model_args = world_model_args
        self.task_name = get_task_name(self.args['env'], self.env_args)
        self.run_dir, self.log_dir, self.save_dir, self.writter \
        = init_dir(
            args["env"], env_args, args["marl_algo"], args["world_model"]
            , args["exp_name"], marl_args["seed"]["seed"], logger_path= marl_args["logger"]["log_dir"] 
        )
        save_config(self.args, self.marl_args, self.world_model_args, self.env_args, self.run_dir)
        self.log_file = open(
             os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        setproctitle.setproctitle(
            str(args["marl_algo"]) + "-" + str(args["world_model"]) + "-" + str(args["env"]) 
        )


        # Attribute regard of agent, env.
        self.envs = make_train_env(
            args["env"],
            marl_args["seed"]["seed"],
            marl_args["train"]["n_rollout_threads"],
            env_args,
        )
        self.eval_envs = make_train_env(
            args["env"],
            marl_args["seed"]["seed"],
            marl_args["train"]["n_rollout_threads"],
            env_args,
        )
        self.num_agents = get_num_agents(args["env"], env_args)
        self.agent_deaths = np.zeros(
            (self.marl_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )
        self.max_action_shape = max([f.n for f in self.envs.action_space])

        # Instance models.

        self.agent = Agent(self.eval_envs, args, marl_args, env_args, world_model_args)
        self.buffer = OffPolicyBufferFP(
            {**marl_args["train"], **marl_args["model"], **marl_args["algo"]},
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.envs.action_space,
            seq_len=self.world_model_args["seq_len"]
        )   


    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
            Args:
                available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                    shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            Returns:
                actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.marl_args["train"]["n_rollout_threads"]):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
        
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)
        return np.array(actions).transpose(1, 0, 2)

    def init_rnns(self):
        self.prev_rnn_state = [None] * self.num_agents
        self.prev_actions = [None] * self.num_agents

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.marl_args["train"]["warmup_steps"]
            // self.marl_args["train"]["n_rollout_threads"]
        )

        obs, share_obs, available_actions = self.envs.reset()
        
        for _ in range(warmup_steps):
            actions = self.sample_actions(available_actions)
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
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
        return obs, share_obs, available_actions


    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True, share_obs=None):
        actions = []
        for agent_id in range(self.num_agents):
            state = self.agent.world_model[agent_id](
                obs[:, agent_id], self.prev_actions[agent_id], self.prev_rnn_state[agent_id]
                )
            feats = state.get_features()
            action = self.agent.actor[agent_id].get_actions(
                        feats,
                        available_actions[:, agent_id],
                        add_random
                    )
            actions.append(
                _t2n(
                    action
                ))
            self.prev_rnn_state[agent_id] = state
            self.prev_actions[agent_id] = F.one_hot(action, num_classes=self.max_action_shape).squeeze(-2) 
        
        return np.array(actions).transpose(1, 0, 2)


    def insert(self, data):
        (
            share_obs, obs, actions, available_actions,  # None or (n_agents, n_threads, action_number)
            rewards, dones, infos,  next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs, next_available_actions
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        
        valid_transitions = 1 - self.agent_deaths
        self.agent_deaths = np.expand_dims(dones, axis=-1)
        terms = np.full(
            (self.marl_args["train"]["n_rollout_threads"], self.num_agents, 1),
            False,
        )
        for i in range(self.marl_args["train"]["n_rollout_threads"]):
            for agent_id in range(self.num_agents):
                if dones[i][agent_id]:
                    if not (
                        "bad_transition" in infos[i][agent_id].keys()
                        and infos[i][agent_id]["bad_transition"] == True
                    ):
                        terms[i][agent_id][0] = True

        for i in range(self.marl_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.marl_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()
        data = (
            share_obs, obs, actions, available_actions, rewards,  # (n_threads, n_agents, 1)
            np.expand_dims(dones, axis=-1), valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
            terms, next_share_obs, next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        )
        self.buffer.insert(data)


    def run(self):
        set_seed(self.marl_args["seed"])
        self.train_episode_rewards = np.zeros(
            self.marl_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        print("-----------------start warmup-----------------")

        obs, share_obs, available_actions = self.warmup()

        print("-----------------start training-----------------")

        steps = (
            self.marl_args["train"]["num_env_steps"]
            // self.marl_args["train"]["n_rollout_threads"]
        )

        update_num = int(  # update number per train
            self.marl_args["train"]["update_per_train"]
        )

        self.init_rnns()
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, available_actions=available_actions, add_random=True,
            )
    
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
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            for i in range(self.marl_args["train"]["n_rollout_threads"]):
                if np.all(dones, axis=1)[i]:    
                    self.init_rnns()

            if step % self.marl_args["train"]["train_interval"] == 0:
                for _ in range(self.world_model_args["model_epochs"]):
                    self.train_world_model()
                for _ in range(update_num): 
                    self.train_agent()

            if step % self.marl_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.marl_args["train"]["warmup_steps"]
                    + step * self.marl_args["train"]["n_rollout_threads"]
                )//self.marl_args["train"]["eval_interval"]
               
                print(
                    f"Env {self.args['env']} Task {self.task_name} Algo {self.args['marl_algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.marl_args['train']['num_env_steps']}:"
                )
                self.eval(cur_step)
    
    def train_world_model(self):
        data = self.buffer.sample_world_batch(self.world_model_args["batch_size"])
        (
            sp_obs, sp_actions, sp_available_actions, sp_reward, sp_done
        ) = data
        for agent_id in range(self.num_agents):
            obs = sp_obs[agent_id]
            actions = sp_actions[agent_id]
            available_actions = sp_available_actions[agent_id]
            reward = sp_reward[agent_id]
            done = sp_done[agent_id]

            self.agent.world_model[agent_id].train()
            self.agent.world_model[agent_id].train_model(
                (obs, actions, available_actions, reward, done)
            )
            self.agent.world_model[agent_id].eval()

    def train_agent(self):
        data = self.buffer.sample_ac_batch()
        self.agent.train_ac(data)

    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.marl_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        eval_battles_won = 0

        episode_lens = []
        one_episode_len = np.zeros(
            self.marl_args["eval"]["n_eval_rollout_threads"], dtype=np.int
        )
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        self.init_rnns()
        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False, 

            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            
            for i in range(self.marl_args["train"]["n_rollout_threads"]):
                if np.all(eval_dones, axis=1)[i]:    
                    self.init_rnns()
            for eval_i in range(self.marl_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.marl_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if "smac" in self.args["env"]:
                        if eval_infos[eval_i][0]["won"]:
                            eval_battles_won += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0


            if eval_episode >= self.marl_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    print(
                        "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_battles_won / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, step
                )

                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, step
                )
                
                break
        # self.eval_envs.close()