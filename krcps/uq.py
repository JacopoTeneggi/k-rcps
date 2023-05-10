import torch
from typing import Tuple
from .utils import register_uq


def _normalize(
    l: torch.Tensor,
    u: torch.Tensor,
    _min: float = 0.0,
    _max: float = 1.0,
    q_eps: float = 1e-06,
) -> Tuple[torch.Tensor, torch.Tensor]:
    l, u = torch.clamp(l, min=_min, max=_max), torch.clamp(u, min=_min, max=_max)
    l[l <= q_eps] = 0.0
    u[u <= q_eps] = 0.0
    return l, u


@register_uq(name="std")
def _std(sampled: torch.Tensor, dim: int = None):
    mu, std = torch.mean(sampled, dim=dim), torch.std(sampled, dim=dim)

    std_min = 1e-02
    std = torch.clamp(std, min=std_min)

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l = mu - _lambda * std
        u = mu + _lambda * std
        return _normalize(l, u)

    return _I


@register_uq(name="naive_sampling_additive")
def _naive_sampling_additive(
    sampled: torch.Tensor, alpha: float = None, dim: int = None
):
    q = torch.quantile(sampled, torch.tensor([alpha / 2, 1 - alpha / 2]), dim=dim)
    l, u = q[0], q[1]

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = l - _lambda
        u_lambda = u + _lambda
        return _normalize(l_lambda, u_lambda)

    return _I


@register_uq(name="naive_sampling_multiplicative")
def _naive_sampling_multiplicative(
    sampled: torch.Tensor, alpha: float = None, dim: int = None
):
    q = torch.quantile(sampled, torch.tensor([alpha / 2, 1 - alpha / 2]), dim=dim)
    l, u = q[0], q[1]

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = l / _lambda
        u_lambda = u * _lambda
        return _normalize(l_lambda, u_lambda)

    return _I


@register_uq(name="calibrated_quantile")
def _calibrated_quantile(sampled: torch.Tensor, alpha: float = None, dim: int = None):
    m = sampled.size(dim)
    q = torch.quantile(
        sampled,
        torch.tensor(
            [
                torch.floor(torch.tensor((m + 1) * alpha / 2)) / m,
                torch.min(
                    torch.tensor(
                        [1, torch.ceil(torch.tensor((m + 1) * (1 - alpha / 2))) / m]
                    )
                ),
            ]
        ),
        dim=dim,
    )
    l, u = q[0], q[1]

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = l - _lambda
        u_lambda = u + _lambda
        return _normalize(l_lambda, u_lambda)

    return _I


@register_uq(name="quantile_regression")
def _quantile_regression(denoised: torch.Tensor):
    l, x, u = denoised[:, 0], denoised[:, 1], denoised[:, 2]

    q_eps = 1e-02
    l, u = torch.clamp(l, min=q_eps), torch.clamp(u, min=q_eps)

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = x - _lambda * l
        u_lambda = x + _lambda * u
        return _normalize(l_lambda, u_lambda)

    return _I


@register_uq(name="conffusion_multiplicative")
def _conffusion_multiplicative(denoised: torch.Tensor):
    _l, _u = denoised[:, 0], denoised[:, 2]
    l, u = torch.minimum(_l, _u), torch.maximum(_l, _u)
    assert torch.all(torch.where(l <= u, 1, 0))

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = l / _lambda
        u_lambda = u * _lambda
        return _normalize(l_lambda, u_lambda)

    return _I


@register_uq(name="conffusion_additive")
def _conffusion_additive(denoised: torch.Tensor):
    _l, _u = denoised[:, 0], denoised[:, 2]
    l, u = torch.minimum(_l, _u), torch.maximum(_l, _u)
    assert torch.all(torch.where(l <= u, 1, 0))

    def _I(_lambda: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_lambda = l - _lambda
        u_lambda = u + _lambda
        return _normalize(l_lambda, u_lambda)

    return _I
