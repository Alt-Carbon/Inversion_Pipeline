"""Regional-residual separation via a 2D polynomial trend surface -- replaces
plain median subtraction. Standard practice for isolating the local anomaly
a bounded 3D mesh can actually explain: the mesh has no way to represent a
deep/large-scale regional field, so that field has to be estimated and
removed from the data before inversion, not left in and expected to be fit
by near-surface density/susceptibility contrasts (see forward.py docstring).
"""
import numpy as np


def fit_polynomial_trend(x, y, v, degree):
    """Least-squares 2D polynomial trend surface, total degree `degree`
    (e.g. degree=2 -> 1, x, y, x^2, xy, y^2). Coordinates are centred and
    scaled before fitting -- required for a well-conditioned fit, since x/y
    are O(1e4) m and higher powers would otherwise span many orders of
    magnitude. Returns (trend, info) where `trend` is the fitted surface
    evaluated at (x, y) and `info` holds what's needed to evaluate the same
    surface elsewhere or explain how much of the data variance it captured.
    """
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    terms = [(i, j) for total in range(degree + 1) for i in range(total + 1) for j in [total - i]]
    A = np.column_stack([xs ** i * ys ** j for i, j in terms])
    coeffs, _, rank, _ = np.linalg.lstsq(A, v, rcond=None)
    trend = A @ coeffs

    resid = v - trend
    var_explained = 1.0 - resid.var() / v.var()
    info = dict(degree=degree, terms=terms, coeffs=coeffs, rank=rank,
               x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
               var_explained=float(var_explained))
    return trend, info
