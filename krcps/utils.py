import numpy as np

_UQ_DICT = {}
_LOSS_DICT = {}
_BOUND_DICT = {}
_MEMBERSHIP_DICT = {}
_CALIBRATION_DICT = {}


def register_fn(dict=None, name=None):
    def _register(func):
        def _fn(*args, **kwargs):
            return func(*args, **kwargs)

        dict[name] = _fn
        return _fn

    return _register


def register_uq(name=None):
    return register_fn(dict=_UQ_DICT, name=name)


def register_loss(name=None):
    return register_fn(dict=_LOSS_DICT, name=name)


def register_bound(name=None):
    return register_fn(dict=_BOUND_DICT, name=name)


def register_membership(name=None):
    return register_fn(dict=_MEMBERSHIP_DICT, name=name)


def register_calibration(name=None):
    return register_fn(dict=_CALIBRATION_DICT, name=name)


def get_uq(name, *args, **kwargs):
    def _f(x, **kfargs):
        return _UQ_DICT[name](x, *args, **kwargs, **kfargs)

    return _f


def get_loss(name, *args, **kwargs):
    def _f(target, l, u, **kfargs):
        return _LOSS_DICT[name](target, l, u, *args, **kwargs, **kfargs)

    return _f


def get_bound(name, *args, **kwargs):
    def _f(n, delta, loss=None, **kfargs):
        return _BOUND_DICT[name](n, delta, loss, *args, **kwargs, **kfargs)

    return _f


def get_membership(name, *args, **kwargs):
    def _f(*fargs, **kfargs):
        return _MEMBERSHIP_DICT[name](*fargs, *args, **kwargs, **kfargs)

    return _f


def get_calibration(name):
    return _CALIBRATION_DICT[name]


def _split_idx(n, n_split):
    split_idx = np.random.choice(n, size=n_split, replace=False).tolist()
    return split_idx, list(set(range(n)) - set(split_idx))


def _set_I(I, set_idx):
    def _f(_lambda):
        l, u = I(_lambda)
        return l[set_idx], u[set_idx]

    return _f
