import torch


def test_register_uq():
    from .utils import register_uq, get_uq

    @register_uq(name="test")
    def _test(x: torch.Tensor):
        return x

    x = torch.Tensor(1)
    assert get_uq("test")(x) == x


def test_register_loss():
    from .utils import register_loss, get_loss

    @register_loss(name="test")
    def _test(target: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
        return target, l, u

    target, l, u = torch.Tensor(1), torch.Tensor(2), torch.Tensor(3)
    assert get_loss("test")(target, l, u) == (target, l, u)


def test_register_bound():
    from .utils import register_bound, get_bound

    @register_bound(name="test")
    def _test(n: int, delta: float, loss: float):
        return n, delta, loss

    n, delta, loss = 1, 2.0, 3.0
    assert get_bound("test")(n, delta, loss) == (n, delta, loss)


def test_split_idx():
    from .utils import _split_idx

    n, n_split = 10, 5
    split_idx, not_split_idx = _split_idx(n, n_split)

    assert len(split_idx) == n_split and len(not_split_idx) == n - n_split
    assert set(split_idx + not_split_idx) == set(range(n))
    assert set(split_idx) & set(not_split_idx) == set()


def test_set_I():
    from .utils import _set_I

    def _I(x: torch.Tensor):
        return x - 1, x + 1

    x = torch.arange(4)
    set_idx = [0, 1]

    expected_l, expected_u = torch.Tensor([-1, 0]), torch.Tensor([1, 2])
    l, u = _set_I(_I, set_idx)(x)
    assert torch.all(l == expected_l) and torch.all(u == expected_u)
