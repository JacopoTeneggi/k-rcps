from typing import Callable

_CONFIGS_DICT = {}
_DATA_CONFIGS_DICT = {}


def register_config(name: str = None) -> Callable:
    def _register(func):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        _CONFIGS_DICT[name] = _fn
        return _fn

    return _register


def register_data_config(name: str = None) -> Callable:
    def _register(func):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        _DATA_CONFIGS_DICT[name] = _fn
        return _fn

    return _register


def get_config(name: str) -> Callable:
    config = _CONFIGS_DICT[name]()
    config.name = name
    return config


def get_data_config(name: str) -> Callable:
    config = _DATA_CONFIGS_DICT[name]()
    config.name = name
    return config
