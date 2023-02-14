import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
from .utils import register_bound


@register_bound(name="hoeffding")
def _hoeffding_bound(n, delta, loss):
    return (loss + np.sqrt(1 / (2 * n) * np.log(1 / delta))).item()


def _hoeffding_plus(r, loss, n):
    h1 = lambda u: u * np.log(u / r) + (1 - u) * np.log((1 - u) / (1 - r))
    return -n * h1(np.maximum(r, loss))


def _bentkus_plus(r, loss, n):
    return np.log(np.maximum(binom.cdf(np.floor(n * loss), n, r), 1e-10)) + 1


@register_bound(name="hoeffding_bentkus")
def _hoeffding_bentkus_bound(n, delta, loss, maxiter=1000):
    def _tailprob(r):
        hoeffding_mu = _hoeffding_plus(r, loss, n)
        bentkus_mu = _bentkus_plus(r, loss, n)
        return np.minimum(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        try:
            return brentq(_tailprob, loss, 1 - 1e-10, maxiter=maxiter)
        except:
            print(f"BRENTQ RUNTIME ERROR at muhat={loss}")
            return 1.0
