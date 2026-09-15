"""Study-area (survey-footprint) outline -- ported from india_grav_mt_
inversion/src/coverage.py unchanged (same distance-to-nearest-station
thresholding approach), for the same reason it was built there: draw the
real receiver coverage, not an implied bounding box, on 2D section
locator panels and in the 3D viewer.

Synthetic400's receivers are a full, dense 400x400 raster with no
interior gaps (unlike WA/India's real, irregular district boundaries),
so applying this here mostly traces the rectangular footprint's own
edge -- a less dramatic result than it produced for WA/India, but a real,
working capability rather than a no-op: if this pipeline is later pointed
at genuinely irregular coverage (e.g. Callisto's real flight-line data,
once that benchmark is run -- see config.py's BENCHMARKS registry), the
same code correctly traces whatever shape (including interior holes) the
receivers actually form, with no changes needed.

Approach: distance-to-nearest-receiver thresholding (not a convex/concave
hull) -- simpler, and it naturally follows the true (possibly non-convex,
possibly multiply-connected) shape without a hull algorithm, including
interior holes. Threshold is `config.COVERAGE_MAX_DIST_M`.
"""
import numpy as np
from scipy.spatial import cKDTree


def coverage_mask(x, y, receiver_x, receiver_y, max_dist):
    """Boolean mask, True where (x, y) is within `max_dist` of some
    receiver."""
    tree = cKDTree(np.c_[receiver_x, receiver_y])
    dist, _ = tree.query(np.c_[x, y])
    return dist <= max_dist


def coverage_contour(receiver_x, receiver_y, max_dist, pad=1000.0, grid_res=250):
    """Boundary polyline(s) of the coverage mask, as a list of (N, 2)
    arrays -- one array per closed loop (coverage with a hole would yield
    2+ loops; Synthetic400's dense rectangular raster yields exactly one,
    tracing its own outer edge). Used to draw the true survey outline on
    2D locator panels and in the 3D viewer, so the shape shown isn't just
    an implied bounding box.

    Extracted via matplotlib's own contour tracing (marching squares) on
    a distance-to-nearest-receiver field, evaluated on a `grid_res` x
    `grid_res` grid -- no figure is displayed or saved, this only uses
    the Path vertices, and works headless (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    x0, x1 = receiver_x.min() - pad, receiver_x.max() + pad
    y0, y1 = receiver_y.min() - pad, receiver_y.max() + pad
    xg = np.linspace(x0, x1, grid_res)
    yg = np.linspace(y0, y1, grid_res)
    XX, YY = np.meshgrid(xg, yg)
    tree = cKDTree(np.c_[receiver_x, receiver_y])
    dist, _ = tree.query(np.c_[XX.ravel(), YY.ravel()])
    dist = dist.reshape(XX.shape)

    fig, ax = plt.subplots()
    cs = ax.contour(XX, YY, dist, levels=[max_dist])
    loops = [seg for seg in cs.allsegs[0] if len(seg) >= 4]
    plt.close(fig)
    return loops
