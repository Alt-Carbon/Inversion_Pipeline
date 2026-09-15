"""Distance-based sensitivity weighting (Li & Oldenburg, 1996b) -- an
alternative to the depth weighting SimPEG's `directives.
UpdateSensitivityWeights` computes by default inside `invert.run_smooth`.

Why this matters once topography is real (Task 1): depth weighting's
usual justification is that a potential-field kernel decays with
distance from a *flat* receiver plane, so "depth below the (flat)
surface" is a good proxy for "distance from the data." Once receivers
follow real topography (draped, not flat -- some over hills, some in
valleys, per src/topography.py), a cell's *depth* below the nearest
patch of ground is no longer the same thing as its *distance* to the
nearest receiver -- a cell just below a valley floor can be much closer
to a receiver than a cell at the same depth-below-local-surface under a
hill. Li & Oldenburg's distance form replaces "depth below surface" with
"distance to nearest receiver" directly, which stays meaningful under
topography.

Formula: w_j = 1 / (r_j + r0)^(beta/2), r_j = distance from active cell j
to its nearest receiver. beta=3 is the standard exponent for magnetic
(dipole-like, ~1/r^3 kernel) data (Li & Oldenburg 1996b use beta=2 for
gravity's ~1/r^2 kernel; this project is magnetics-only, so beta=3 is the
default and the only case exercised here). r0 is a small stabilisation
constant preventing a blow-up for cells essentially at a receiver;
Li & Oldenburg fit it against the actual kernel's own decay in their
paper -- that fit isn't reproduced here (would need the full analytic
kernel-decay derivation, out of scope for a "toggle this weighting
scheme on" round), so r0 defaults to core_cell_z_m (the mesh's own
finest vertical resolution), a documented simplification, not a
rigorously calibrated value -- flagged in improved_report.md.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def distance_weights(mesh, active_cells, receiver_locations, beta=3.0, r0=None,
                     core_cell_z_m=None):
    """Per-active-cell weight array, normalised to max=1 (same convention
    SimPEG's own UpdateSensitivityWeights uses for its depth weights, so
    the two are on a comparable scale when swapped in via `weights=`)."""
    if r0 is None:
        if core_cell_z_m is None:
            raise ValueError("pass r0 explicitly or core_cell_z_m to default r0 from it")
        r0 = core_cell_z_m
    cc = mesh.cell_centers[active_cells]
    tree = cKDTree(receiver_locations)
    r, _ = tree.query(cc, k=1)
    w = 1.0 / (r + r0) ** (beta / 2.0)
    return w / w.max()


def fit_r0(mesh, active_cells, receiver_locations, G, beta=3.0, r0_bounds=(1.0, 5000.0)):
    """Calibrates r0 the way Li & Oldenburg (1996b) actually calibrate
    their weighting function -- fit it against the REAL sensitivity decay
    of this forward operator, not a rule-of-thumb (core_cell_z_m, what
    `distance_weights` falls back to without this). Needs the actual G
    matrix (n_data x n_active_cells) -- expensive to compute (~30-70 s at
    this project's resolution) but only needs computing once per geometry;
    pair with src/kernel_cache.py to make repeated calibration/reruns
    cheap.

    Method: per-cell RMS column sensitivity of G is the ground truth for
    "how much does this cell actually influence the data" -- the same
    quantity SimPEG's own `directives.UpdateSensitivityWeights` uses
    directly (see its docstring, read from source). The analytic distance
    form `w(r)=1/(r+r0)^(beta/2)` is a cheap proxy for that -- r0 is fit
    (bounded 1-sided least squares on the normalised curves, both scaled
    to max=1) so the proxy matches the real thing as closely as a single
    scalar can, rather than guessing r0's order of magnitude from the
    mesh's own cell size."""
    cc = mesh.cell_centers[active_cells]
    tree = cKDTree(receiver_locations)
    r, _ = tree.query(cc, k=1)

    rms_sensitivity = np.sqrt(np.mean(np.asarray(G) ** 2, axis=0))
    target = rms_sensitivity / rms_sensitivity.max()

    def loss(r0):
        w = 1.0 / (r + r0) ** (beta / 2.0)
        w = w / w.max()
        return float(np.sum((w - target) ** 2))

    result = minimize_scalar(loss, bounds=r0_bounds, method="bounded")
    return float(result.x), dict(loss=float(result.fun), r0_bounds=r0_bounds,
                                 hit_lower_bound=abs(result.x - r0_bounds[0]) < 1e-3,
                                 hit_upper_bound=abs(result.x - r0_bounds[1]) < 1e-3)
