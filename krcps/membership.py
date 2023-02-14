import torch
import numpy as np
from skimage.filters import threshold_multiotsu
from .utils import get_loss, register_membership


@register_membership(name="01_loss_quantile")
def _01_loss_quantile(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    q = torch.quantile(loss.view(-1), torch.arange(0, 1, 1 / k)[1:]).unique()
    k = len(q) + 1

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _q in enumerate(reversed(q)):
        m[loss <= _q] = k - (i + 2)

    qcoords = []
    for _k in range(k):
        qcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(qcoords) == len(q) + 1 == k
    assert all([len(_q[0]) == len(_q[1]) for _q in qcoords])
    assert sum([len(_q[0]) for _q in qcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _q in enumerate(qcoords):
        nk[_k] = len(_q[0])
        m[_q[0], _q[1], _k] = 1
    return k, nk, m


@register_membership(name="01_loss_otsu")
def _01_loss_otsu(opt_set, opt_l, opt_u, k):
    loss_fn = get_loss("vector_01")
    loss = loss_fn(opt_set, opt_l, opt_u)

    t = threshold_multiotsu(loss.numpy(), classes=k)
    k = len(t) + 1

    m = (k - 1) * torch.ones_like(loss, dtype=torch.long)
    for i, _t in enumerate(reversed(t)):
        m[loss <= _t] = k - (i + 2)

    tcoords = []
    for _k in range(k):
        tcoords.append(torch.nonzero(m == _k, as_tuple=True))

    assert len(tcoords) == len(t) + 1 == k
    assert all([len(_t[0]) == len(_t[1]) for _t in tcoords])
    assert sum([len(_t[0]) for _t in tcoords]) == torch.numel(loss)

    nk = np.empty((k))
    m = torch.zeros(opt_set.size(-2), opt_set.size(-1), k)
    for _k, _t in enumerate(tcoords):
        nk[_k] = len(_t[0])
        m[_t[0], _t[1], _k] = 1
    return k, nk, m
