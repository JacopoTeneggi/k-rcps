import numpy as np
import torch
from typing import Callable, List, Tuple

_UQ_DICT = {}
_LOSS_DICT = {}
_BOUND_DICT = {}
_MEMBERSHIP_DICT = {}
_CALIBRATION_DICT = {}


def register_fn(dict: dict = None, name: str = None) -> Callable:
    def _register(func: Callable):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        dict[name] = _fn
        return _fn

    return _register


def register_uq(name: str = None) -> Callable:
    return register_fn(dict=_UQ_DICT, name=name)


def register_loss(name: str = None) -> Callable:
    return register_fn(dict=_LOSS_DICT, name=name)


def register_bound(name: str = None) -> Callable:
    return register_fn(dict=_BOUND_DICT, name=name)


def register_membership(name: str = None) -> Callable:
    return register_fn(dict=_MEMBERSHIP_DICT, name=name)


def register_calibration(name: str = None) -> Callable:
    return register_fn(dict=_CALIBRATION_DICT, name=name)


def get_uq(name: str, *args, **kwargs) -> Callable:
    def _f(x: torch.Tensor, **kfargs):
        return _UQ_DICT[name](x, *args, **kwargs, **kfargs)

    return _f


def get_loss(name: str, *args, **kwargs) -> Callable:
    def _f(target: torch.Tensor, l: torch.Tensor, u: torch.Tensor, **kfargs):
        return _LOSS_DICT[name](target, l, u, *args, **kwargs, **kfargs)

    return _f


def get_bound(name: str, *args, **kwargs) -> Callable:
    def _f(n: int, delta: float, loss: float = None, **kfargs):
        return _BOUND_DICT[name](n, delta, loss, *args, **kwargs, **kfargs)

    return _f


def get_membership(name: str, *args, **kwargs) -> Callable:
    def _f(set: torch.Tensor, l: torch.Tensor, u: torch.Tensor, k: int, **kfargs):
        return _MEMBERSHIP_DICT[name](set, l, u, k, *args, **kwargs, **kfargs)

    return _f


def get_calibration(name: str) -> Callable:
    return _CALIBRATION_DICT[name]


def _split_idx(n: int, n_split: int) -> Tuple[List[int], List[int]]:
    split_idx = np.random.choice(n, size=n_split, replace=False).tolist()
    return split_idx, list(set(range(n)) - set(split_idx))


def _set_I(I: Callable, set_idx: List[int]) -> Callable:
    def _f(_lambda: torch.Tensor):
        l, u = I(_lambda)
        return l[set_idx], u[set_idx]

    return _f
