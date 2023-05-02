import torch
from pytest import raises


def test_01_vector_loss():
    from .losses import _01_vector_loss

    l, u = 0.25, 0.75
    target = torch.linspace(0, 1, 10)

    reduction = "unknown reduction"
    with raises(ValueError):
        _01_vector_loss(target, l, u, reduction=reduction)

    reduction = "none"
    loss = _01_vector_loss(target, l, u, reduction=reduction)
    expected_loss = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    assert torch.all(loss == expected_loss)

    reduction = "mean"
    loss = _01_vector_loss(target, l, u, reduction=reduction, dim=0)
    expected_loss = 0.6
    assert loss == expected_loss


def test_01_loss():
    from .losses import _01_loss

    l, u = 0.25, 0.75
    target = torch.linspace(0, 1, 10)

    loss = _01_loss(target, l, u)
    expected_loss = 0.6
    assert loss == expected_loss


def test_gamma_vector_loss():
    from .losses import _gamma_vector_loss

    l, u = 0.25, 0.75
    c = (l + u) / 2
    i = u - l
    target = torch.linspace(0, 1, 10)

    gamma = -1
    with raises(ValueError):
        _gamma_vector_loss(target, l, u, gamma=gamma)

    gamma = 1
    with raises(ValueError):
        _gamma_vector_loss(target, l, u, gamma=gamma)

    reduction = "unknown reduction"
    with raises(ValueError):
        _gamma_vector_loss(target, l, u, reduction=reduction)

    reduction = "none"
    gamma = 0
    loss = _gamma_vector_loss(target, l, u, gamma=gamma, reduction=reduction)
    expected_loss = 2 * torch.abs(target - c) / i
    assert torch.all(loss == expected_loss)

    reduction = "none"
    gamma = 0.5
    loss = _gamma_vector_loss(target, l, u, gamma=gamma, reduction=reduction)
    assert torch.all(loss[torch.logical_and(target < l, target > u)] >= 1)
    assert torch.all(
        loss[
            torch.logical_and(target >= c - gamma / 2 * i, target <= c + gamma / 2 * i)
        ]
        == 0
    )

    reduction = "none"
    gamma = 0.9999
    loss = _gamma_vector_loss(target, l, u, gamma=gamma, reduction=reduction)
    assert torch.all(loss[torch.logical_and(target >= l, target <= u)] == 0)

    reduction = "mean"
    gamma = 0
    loss = _gamma_vector_loss(target, l, u, gamma=gamma, reduction=reduction, dim=0)
