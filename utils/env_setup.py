
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def get_num_agents(env_name, env_args):
    """Get the number of agents in the environment."""
    if env_name == "smac":
        from envs.smac.smac_maps import get_map_params

        return get_map_params(env_args["map_name"])["n_agents"]
    

def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
           
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])
