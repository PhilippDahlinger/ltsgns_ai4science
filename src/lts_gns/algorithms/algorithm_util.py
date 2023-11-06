from typing import Iterable, List, Any


def list_from_config(config_parameter: Iterable[Any] | Any | None) -> List[Any]:
    if config_parameter is None:
        config_parameter = []
    elif isinstance(config_parameter, str) or not isinstance(config_parameter, Iterable):
        config_parameter = [config_parameter]
    else:
        config_parameter = list(config_parameter)
    return config_parameter
