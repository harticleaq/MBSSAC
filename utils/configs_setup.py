""" Experiment prepared configs """
import time
import os
import json
import yaml
from uu import Error


def get_defaults_yaml_args(*args):
    """Load config file for user-specified algo and env.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    all_args = []
    for arg in args:
        arg_path = os.path.join(base_path, "configs", f"{arg}.yaml")
        with open(arg_path, "r", encoding="utf-8") as file:
            file_arg = yaml.load(file, Loader=yaml.FullLoader)
            all_args.append(file_arg)
    return all_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)