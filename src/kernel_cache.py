"""Sensitivity-kernel (G matrix) caching across separate runs/processes --
for sweeping regularization/beta/reference-model choices against the same
forward geometry (Task 3). Built entirely on SimPEG's own public,
documented caching path (`store_sensitivities="disk"` + `sensitivity_path`
-- verified directly from source, not guessed: `magnetics.Simulation3D
Integral.G` calls `self.linear_operator()` on first access, which itself
checks `sensitivity_path/sensitivity.npy` and mmap-loads it if the shape
matches, before falling back to actually computing it -- see
`linear_operator`'s source in simpeg/potential_fields/base.py). No
`WeightedLeastSquares`/`InexactGaussNewton` code is touched, and no
private `sim._G` poking is needed -- this only sets two already-public
simulation attributes.

Why a hash-keyed directory, not just SimPEG's own path-based caching
directly: `linear_operator`'s own cache-hit check is shape-only
(`kernel.shape == (nD, n_cells)`) -- it will happily load a *wrong*
cached matrix from a stale directory if two different geometries happen
to produce the same shape (e.g. two runs with the same decimate_stride
and core_cell_m but a different IGRF inclination). Keying the directory
name by a content hash of everything that actually determines G makes
that collision essentially impossible, so SimPEG's shape check becomes a
safety net rather than the only thing preventing a silently wrong reuse.
"""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "kernel_cache"


def cache_key(mesh, active_cells, receiver_locations, F, I, D, component="tmi"):
    """Hash of everything that determines G, and nothing that doesn't --
    NOT regularization, beta, reference model, or noise/std (those affect
    the inversion, not the forward geometry). Active-cell *centres*
    (not just the count) are hashed, so two runs with the same number of
    active cells but different topography draping/positions still get
    different keys."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.cell_centers[active_cells]).tobytes())
    h.update(np.ascontiguousarray(receiver_locations).tobytes())
    h.update(np.array([F, I, D], dtype=np.float64).tobytes())
    h.update(component.encode())
    return h.hexdigest()[:20]


def attach_cache(sim, key, cache_dir=DEFAULT_CACHE_DIR):
    """Points `sim` (an already-constructed, not-yet-evaluated
    Simulation3DIntegral) at a hash-keyed disk cache directory for its
    sensitivity matrix. Returns (sim, cache_dir_for_key, is_new) --
    is_new=True means no cached file exists yet at this path (the
    upcoming G computation will be the one that populates it); False
    means a matching-shape cached kernel is already there and will be
    mmap-loaded instead of recomputed. Must be called BEFORE the sim's
    G/fields/dpred is ever accessed -- SimPEG checks the cache file only
    on first access to `sim.G` (see module docstring)."""
    path = Path(cache_dir) / key
    path.mkdir(parents=True, exist_ok=True)
    is_new = not (path / "sensitivity.npy").exists()
    sim.store_sensitivities = "disk"
    sim.sensitivity_path = str(path) + "/"
    return sim, path, is_new


def clear_cache(cache_dir=DEFAULT_CACHE_DIR):
    """Removes every cached kernel under cache_dir. Not called
    automatically anywhere -- kernels don't go stale on their own (the
    hash changes if the geometry does), this is just for reclaiming disk
    space."""
    import shutil
    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)
