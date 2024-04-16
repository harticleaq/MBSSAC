# -*- coding: utf-8 -*-
"""
@Author: Anqi Huang
@Time: 2024/4/10
"""


import argparse
import os
import sys
from utils.configs_setup import get_defaults_yaml_args, update_args
sys.path.append("")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--marl_algo",
        type=str,
        default="ossmc",
        help="Algorithm name.",
    )
    parser.add_argument(
        "--world_model",
        type=str,
        default="dreamer",
        help="Algorithm name.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="smac",
        help="Environment name.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="marl-based", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
   
    marl_args, env_args, wm_args = get_defaults_yaml_args(args["marl_algo"], args["env"], args["world_model"])
    update_args(unparsed_dict, marl_args, env_args, wm_args)  # update args from command line
    
    # revise marl args
    wm_args["stochastic_size"] = wm_args["n_classes"] * wm_args["n_categoricals"]
    wm_args["feat_size"] = wm_args["stochastic_size"] + wm_args["deterministic_size"]
    wm_args["global_feat"] = wm_args["feat_size"] + wm_args["embed_size"]

    # start training
    from runners.runner import Runner

    marl_args["logger"]["log_dir"] = os.path.join(os.path.dirname(__file__), 'loggers') 
    runner = Runner(args, marl_args, env_args, wm_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
