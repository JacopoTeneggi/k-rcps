import os
import torch
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from .utils import (
    get_loss,
    get_bound,
    get_membership,
    register_calibration,
    _split_idx,
    _set_I,
)
from typing import Iterable, Callable


def _rcps(
    rcps_set: torch.Tensor,
    I: Callable,
    loss_name: str,
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: torch.Tensor,
    stepsize: float,
    eta: torch.Tensor = None,
):
    loss_fn = get_loss(loss_name)
    bound_fn = get_bound(bound_name)

    n_rcps = rcps_set.size(0)

    _lambda = lambda_max
    if eta is None:
        eta = torch.ones_like(_lambda)

    loss = loss_fn(rcps_set, *I(_lambda))
    ucb = bound_fn(n_rcps, delta, loss)

    pbar = tqdm(total=epsilon)
    pbar.update(ucb)
    pold = ucb

    while ucb <= epsilon:
        pbar.update(ucb - pold)
        pold = ucb

        prev_lambda = _lambda.clone()
        if torch.all(prev_lambda == 0):
            break

        _lambda -= stepsize * eta
        _lambda = torch.clamp(_lambda, min=0)

        loss = loss_fn(rcps_set, *I(_lambda))
        ucb = bound_fn(n_rcps, delta, loss)
    _lambda = prev_lambda

    pbar.update(epsilon - pold)
    pbar.close()
    return _lambda


@register_calibration(name="rcps")
def _calibrate_rcps(
    cal_set: torch.Tensor,
    I: Callable[[torch.Tensor], torch.Tensor],
    loss_name: str,
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: float,
    stepsize: float,
):
    lambda_max = torch.tensor(lambda_max)
    _lambda = _rcps(
        cal_set, I, loss_name, bound_name, epsilon, delta, lambda_max, stepsize
    )
    return _lambda


def _gamma_loss_fn(i, offset, q, _lambda):
    i_lambda = i + 2 * _lambda
    inv_i_lambda = cp.multiply(cp.inv_pos(i_lambda), offset)
    loss = 2 * (1 + q) * inv_i_lambda - q
    loss = cp.pos(loss)
    return loss


def _pk(opt_set, opt_I, epsilon, lambda_max, k, membership_name, prob_size):
    n_opt = opt_set.size(0)
    opt_l, opt_u = opt_I(0)

    membership_fn = get_membership(membership_name)
    k, nk, m = membership_fn(opt_set, opt_l, opt_u, k)

    d = np.prod(opt_set.size()[-2:])
    prob_nk = np.round(prob_size / d * nk).astype(int)
    prob_i, prob_j, prob_lambda = [], [], []
    for _k, _nk in enumerate(prob_nk):
        _ki, _kj = torch.nonzero(m[:, :, _k] == 1, as_tuple=True)
        _kidx = np.random.choice(
            torch.sum(m[:, :, _k]).long().item(), size=_nk, replace=False
        )
        prob_i.extend(_ki[_kidx])
        prob_j.extend(_kj[_kidx])
        prob_lambda.extend(_nk * [_k])

    _lambda = cp.Variable(k)
    q = cp.Parameter(nonneg=True)

    c = (opt_l + opt_u) / 2
    i = opt_u - opt_l
    offset = torch.abs(opt_set - c)
    i_npy, offset_npy = i.numpy(), offset.numpy()
    r_hat = cp.sum(
        _gamma_loss_fn(
            i_npy[:, prob_i, prob_j],
            offset_npy[:, prob_i, prob_j],
            q,
            _lambda[[prob_lambda]],
        )
    ) / (n_opt * np.sum(prob_nk))

    obj = cp.Minimize(cp.sum(cp.multiply(prob_nk, _lambda)))
    constraints = [_lambda >= 0, _lambda <= lambda_max, r_hat <= epsilon]
    pk = cp.Problem(obj, constraints)
    return (pk, q, _lambda), m


@register_calibration(name="k_rcps")
def _calibrate_k_rcps(
    cal_set: torch.Tensor,
    I: Callable[[torch.Tensor], torch.Tensor],
    bound_name: str,
    epsilon: float,
    delta: float,
    lambda_max: float,
    stepsize: torch.Tensor,
    k: int,
    membership_name: str,
    n_opt: int,
    prob_size: float,
    gamma: Iterable[float],
):
    n = cal_set.size(0)
    opt_idx, rcps_idx = _split_idx(n, n_opt)

    opt_set = cal_set[opt_idx]
    rcps_set = cal_set[rcps_idx]

    opt_I = _set_I(I, opt_idx)
    rcps_I = _set_I(I, rcps_idx)

    prob, m = _pk(opt_set, opt_I, epsilon, lambda_max, k, membership_name, prob_size)
    pk, q, _lambda = prob

    def _solve(gamma):
        q.value = gamma / (1 - gamma)

        if os.path.exists(os.path.expanduser("~/mosek/mosek.lic")):
            pk.solve(
                solver=cp.MOSEK,
                verbose=False,
                warm_start=True,
                mosek_params={"MSK_IPAR_NUM_THREADS": 1},
            )
        else:
            pk.solve(verbose=False, warm_start=True)
        lambda_k, obj = torch.tensor(_lambda.value, dtype=torch.float32), pk.value
        return lambda_k, obj

    sol = [_solve(_gamma) for _gamma in tqdm(gamma)]
    sol = sorted(sol, key=lambda x: x[-1])
    lambda_k, _ = sol[0]

    _lambda = torch.matmul(m, lambda_k) + lambda_max
    _lambda = _rcps(
        rcps_set, rcps_I, "01", bound_name, epsilon, delta, _lambda, stepsize
    )
    return _lambda
