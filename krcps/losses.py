import torch
import cvxpy as cp
from .utils import register_loss


@register_loss(name="vector_01")
def _01_vector_loss(target, l, u, reduction="mean", dim=0):
    loss = torch.where(torch.logical_and(target >= l, target <= u), 0.0, 1.0)
    if reduction == "mean":
        return torch.mean(loss, dim=dim)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction}")


@register_loss(name="01")
def _01_loss(target, l, u):
    vector_loss = _01_vector_loss(target, l, u)
    return torch.mean(vector_loss)


@register_loss(name="vector_gamma")
def _gamma_vector_loss(target, l, u, gamma=0.5):
    pass
