import torch
from .utils import register_uq


@register_uq(name="calibrated_quantile")
def _calibrated_quantile(sampled, alpha=None, dim=None):
    n = sampled.size(dim)
    q = torch.quantile(
        sampled,
        torch.tensor(
            [
                torch.floor(torch.tensor((n + 1) * alpha / 2)) / n,
                torch.min(
                    torch.tensor(
                        [1, torch.ceil(torch.tensor((n + 1) * (1 - alpha / 2))) / n]
                    )
                ),
            ]
        ),
        dim=dim,
    )
    l, u = q[0], q[1]

    def _I(_lambda):
        l_lambda = l - _lambda
        u_lambda = u + _lambda
        return l_lambda, u_lambda

    return _I


@register_uq(name="quantile_regression")
def _quantile_regression(denoised):
    l, x, u = denoised[:, 0], denoised[:, 1], denoised[:, 2]

    q_eps = 1e-02
    l, u = torch.clamp(l, min=q_eps), torch.clamp(u, min=q_eps)

    def _I(_lambda):
        l_lambda = x - _lambda * l
        u_lambda = x + _lambda * u
        return l_lambda, u_lambda

    return _I
