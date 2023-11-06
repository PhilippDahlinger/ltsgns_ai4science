from typing import List, Any, Dict, Union

import numpy as np

from lts_gns.util.config_dict import ConfigDict
from lts_gns.util.own_types import ValueDict


def prefix_keys(dictionary: Dict[str, Any], prefix: Union[str, List[str]], separator: str = "/") -> Dict[str, Any]:
    if isinstance(prefix, str):
        prefix = [prefix]
    prefix = separator.join(prefix + [""])
    return {prefix + k: v for k, v in dictionary.items()}


def add_to_dictionary(dictionary: ValueDict, new_scalars: ValueDict) -> ValueDict:
    for k, v in new_scalars.items():
        if k not in dictionary:
            dictionary[k] = []
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim == 1):
            dictionary[k] = dictionary[k] + v
        else:
            dictionary[k].append(v)
    return dictionary


def deep_update(mapping: ValueDict, *updating_mappings: ValueDict) -> ValueDict:
    """
    Update a mapping recursively. If a key is present in both mappings, the value of the updating mapping is used.
    :param mapping:
    :param updating_mappings:
    :return:
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping

def deep_update_weird_custom_config_dicts(weird_config_dict1: ConfigDict, weird_config_dict2: ConfigDict) -> ConfigDict:
    """
    Deep Update for weird custom config dicts, because apparently the normal deep_update function doesn't work for them
    because they do not interface the copy() method (and can not easily do so for some reason).
    Args:
        weird_config_dict1:
        weird_config_dict2:

    Returns: A recursively updated config dict.

    """
    raw_dict1 = weird_config_dict1.get_raw_dict()
    raw_dict2 = weird_config_dict2.get_raw_dict()
    updated_raw_dict = deep_update(raw_dict1, raw_dict2)
    return ConfigDict.from_python_dict(updated_raw_dict)

def check_against_default(config_dict: ConfigDict, default_config_dict: ConfigDict, allowed_exceptions: list[str], root_key=None) -> bool:
    """
    Checks whether the config_dict has the same keys as the default_config_dict. If not, it prints the differences.
    :param config_dict:
    :param default_config_dict:
    :param allowed_exceptions: Keys that are allowed to be missing in the default_config_dict.
    :param root_key: The root key of the config_dict. Used for printing.
    :return: True if the config_dict has the same keys as the default_config_dict, False otherwise.
    """
    result = True
    for k, v in config_dict.items():
        # allow exceptions
        if root_key is not None:
            if f"{root_key}.{k}" in allowed_exceptions:
                continue
        else:
            if k in allowed_exceptions:
                continue
        if k not in default_config_dict:
            if root_key is not None:
                print(f"Key '{root_key}.{k}' is not in default config dict")
            else:
                print(f"Key '{k}' is not in default config dict")
            print("Valid keys are:")
            for k in default_config_dict.keys():
                print("  " + k)
            result = False
        elif isinstance(v, dict):
            if root_key is not None:
                result = result and check_against_default(config_dict[k], default_config_dict[k], allowed_exceptions=allowed_exceptions,
                                                          root_key=f"{root_key}.{k}")
            else:
                result = result and check_against_default(config_dict[k], default_config_dict[k], allowed_exceptions=allowed_exceptions,
                                                          root_key=k)
    return result
