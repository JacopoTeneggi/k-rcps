import torch


def test_calibrated_quantile():
    from .uq import _calibrated_quantile

    m = 128
    sampled = torch.arange(start=1, end=m + 1).float()
    print(len(sampled))

    alpha = 0.10
    I = _calibrated_quantile(sampled, alpha=alpha, dim=0)

    l0, u0 = I(0)
    expected_l0, expected_u0 = 6, 123
    assert (
        len(sampled[sampled <= l0]) == expected_l0
        and len(sampled[sampled <= u0]) == expected_u0
    )

    l, u = I(1)
    expected_l, expected_u = 5, 124
    assert (
        len(sampled[sampled <= l]) == expected_l
        and len(sampled[sampled <= u]) == expected_u
    )


def test_quantile_regression():
    from .uq import _quantile_regression

    x = torch.arange(4)
    l = u = torch.ones_like(x)
    denoised = torch.stack([l, x, u], dim=1)

    I = _quantile_regression(denoised)

    l0, u0 = I(0)
    expected_l0 = expected_u0 = x
    assert torch.all(l0 == expected_l0) and torch.all(u0 == expected_u0)

    l, u = I(0.1)
    expected_l, expected_u = x - 0.1, x + 0.1
    assert torch.all(l == expected_l) and torch.all(u == expected_u)

    l = u = torch.zeros_like(x)
    denoised = torch.stack([l, x, u], dim=1)

    I = _quantile_regression(denoised)

    l, u = I(1)
    expected_l, expected_u = x - 1e-02, x + 1e-02
    assert torch.all(l == expected_l) and torch.all(u == expected_u)
