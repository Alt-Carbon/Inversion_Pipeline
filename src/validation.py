"""Recovery metrics for the Synthetic400 baseline run, checked against
Ogarko et al. (2024)'s six synthetic bodies.

Ground truth availability -- checked directly, not assumed (see
load_survey.py's module docstring and baseline_report.md's "Data
and Ground Truth" section for the full trail):
  - Susceptibility contrast per body: exact, from the paper's Table 3
    (downloaded as XLSX directly, not read off the rendered table image).
  - Shape per body: exact, same source, cross-checked against Fig. 4's
    outlines (disc/cylinder bodies render as circles in map view, slabs
    as rectangles -- consistent between the two independent sources).
  - Horizontal centroid: NOT given anywhere as a number. Digitised by eye
    from Fig. 4 (map-view total response with body outlines on a 2500 m
    graticule), using the receiver footprint's own extent to anchor the
    graticule to real coordinates (see BODIES' `centroid_source` field).
    This is a rough visual reading, not a pixel-precise measurement --
    treat the `centroid_uncertainty_m` on each body as a real error bar,
    not a formality.
  - Depth-to-top: NOT available from any source found (not in the two
    tables, not in the article text, no cross-section of the *true*
    model is published -- only Fig. 6, which shows the *inverted*
    result). No numeric depth-error metric is computed here; depth
    recovery is reported qualitatively only (which depth band the
    recovered peak falls in).
"""
import numpy as np

# Core box used to anchor Fig. 4's graticule to real coordinates: centred
# on the receiver footprint's own midpoint (154992, 9794991.5 -- from
# MAG_n5f1.OBS, X 150004.5-159979.5, Y 9790004-9799979), rounded out to
# the paper's own stated "10 km x 10 km central core" (Table 2). Off by
# at most ~12.5 m from the receivers' literal bounding box, negligible
# next to the graticule-reading uncertainty below.
CORE_X0, CORE_X1 = 149992.0, 159992.0
CORE_Y0, CORE_Y1 = 9789992.0, 9799992.0
GRATICULE_M = 2500.0

# id, shape/susceptibility from Table 3 (XLSX, exact); centroid digitised
# from Fig. 4 by eye in graticule-cell units (col, row), col counted from
# the west edge, row from the *north* edge (image top = north) --
# converted to metres via CORE_X0/Y1 + col/row * GRATICULE_M.
# centroid_uncertainty_m is a rough reading-error estimate (roughly half
# a graticule cell), not a rigorously derived figure.
BODIES = [
    dict(id=1, shape="flat disc", susceptibility_si=0.05,
        col=1.67, row=2.50, centroid_uncertainty_m=350,
        centroid_source="Fig.4 map-view outline, digitised by eye"),
    dict(id=2, shape="flat slab", susceptibility_si=0.07,
        col=0.50, row=3.02, centroid_uncertainty_m=500,
        centroid_source="Fig.4 -- body extends past the image's west "
                        "edge, so this reading likely undershoots the "
                        "true centroid to the east; larger uncertainty"),
    dict(id=3, shape="dipping slab", susceptibility_si=0.03,
        col=2.48, row=2.58, centroid_uncertainty_m=350,
        centroid_source="Fig.4 map-view outline, digitised by eye"),
    dict(id=4, shape="vertical cylinder", susceptibility_si=0.10,
        col=1.60, row=3.90, centroid_uncertainty_m=300,
        centroid_source="Fig.4 map-view outline, digitised by eye"),
    dict(id=5, shape="dipping slab", susceptibility_si=0.02,
        col=0.90, row=1.00, centroid_uncertainty_m=350,
        centroid_source="Fig.4 map-view outline, digitised by eye"),
    dict(id=6, shape="vertical cylinder", susceptibility_si=0.0015,
        col=3.63, row=1.58, centroid_uncertainty_m=350,
        centroid_source="Fig.4 map-view outline, digitised by eye -- "
                        "NOTE: true value (0.0015 SI) is below the "
                        "0.01 SI contour used for volume comparison, so "
                        "this body cannot appear in that metric even "
                        "under perfect recovery -- see report"),
]
for _b in BODIES:
    _b["true_xy"] = (CORE_X0 + _b["col"] * GRATICULE_M, CORE_Y1 - _b["row"] * GRATICULE_M)


def recovered_blob_near(cc, model, true_xy, search_radius_m=1500.0, expected_sign=None):
    """Within `search_radius_m` of the (approximate) true horizontal
    centroid, in any column depth: the recovered model's peak |value|,
    the cell it occurs at, and a value-weighted horizontal centroid of
    cells above 1% of that local peak (a compact summary of "is there a
    recovered anomaly here at all, and if so where/how strong").

    `expected_sign` (default None -- picks the largest |value| regardless
    of sign, round 1's original behaviour, kept as the default so
    baseline_report.md's numbers stay reproducible): pass +1 or -1 to
    instead pick the largest value of THAT sign specifically. Added for
    round 2 -- see improved_report.md's sign-check finding: once distance
    weighting made opposite-sign side-lobes near some bodies larger than
    the body's own correctly-signed peak, sign-agnostic argmax(|value|)
    started picking up the artifact instead of the body, which is a
    real, reportable finding in its own right, not something to paper
    over by silently switching the default -- callers doing a "how well
    was body X itself recovered" comparison should pass expected_sign
    explicitly and report the sign-agnostic peak alongside it (see
    per_body_report)."""
    dx = cc[:, 0] - true_xy[0]
    dy = cc[:, 1] - true_xy[1]
    within = (dx * dx + dy * dy) <= search_radius_m ** 2
    if not within.any():
        return None
    vals = model[within]
    ccw = cc[within]
    if expected_sign is None:
        i_peak = np.argmax(np.abs(vals))
    else:
        signed = vals * expected_sign
        i_peak = np.argmax(signed)
        if signed[i_peak] <= 0:
            return None   # no cell of the expected sign at all within the search radius
    peak_val = vals[i_peak]
    peak_xyz = ccw[i_peak]
    weight_mask = np.abs(vals) >= 0.01 * abs(peak_val) if peak_val != 0 else np.zeros_like(vals, bool)
    if weight_mask.sum() >= 3:
        w = np.abs(vals[weight_mask])
        cx = np.average(ccw[weight_mask, 0], weights=w)
        cy = np.average(ccw[weight_mask, 1], weights=w)
    else:
        cx, cy = peak_xyz[0], peak_xyz[1]
    return dict(peak_value=float(peak_val), peak_xyz=tuple(peak_xyz.tolist()),
               weighted_centroid_xy=(float(cx), float(cy)),
               n_cells_searched=int(within.sum()))


def contour_volume(cc, model, cell_volumes, contour_si=0.01, near_xy=None, search_radius_m=1500.0):
    """Total volume (m^3) of active cells with model value >= contour_si,
    optionally restricted to a horizontal neighbourhood of `near_xy` (so
    one body's volume isn't inflated by an unrelated body's cells
    elsewhere in the mesh)."""
    mask = model >= contour_si
    if near_xy is not None:
        dx = cc[:, 0] - near_xy[0]
        dy = cc[:, 1] - near_xy[1]
        mask &= (dx * dx + dy * dy) <= search_radius_m ** 2
    return float(cell_volumes[mask].sum()), int(mask.sum())


def depth_band(cc, model, near_xy, search_radius_m=1500.0, value_threshold_frac=0.5):
    """Qualitative depth recovery only (no numeric ground truth exists --
    see module docstring): the z-range spanned by cells near `near_xy`
    whose |value| is at least `value_threshold_frac` of the local peak."""
    dx = cc[:, 0] - near_xy[0]
    dy = cc[:, 1] - near_xy[1]
    within = (dx * dx + dy * dy) <= search_radius_m ** 2
    if not within.any():
        return None
    vals = np.abs(model[within])
    z = cc[within, 2]
    peak = vals.max()
    sig = z[vals >= value_threshold_frac * peak]
    return float(sig.max()), float(sig.min())   # (shallowest, deepest) z of significant recovery


def per_body_report(cc, model, cell_volumes, search_radius_m=1500.0):
    """Runs recovered_blob_near / contour_volume / depth_band for every
    body in BODIES; returns one dict per body with everything
    baseline_report.md's table needs, plus explicit uncertainty fields so
    the report can't accidentally present these as exact."""
    rows = []
    for b in BODIES:
        true_sign = 1 if b["susceptibility_si"] >= 0 else -1
        blob = recovered_blob_near(cc, model, b["true_xy"], search_radius_m)
        same_sign_blob = recovered_blob_near(cc, model, b["true_xy"], search_radius_m,
                                             expected_sign=true_sign)
        vol_m3, n_vox = contour_volume(cc, model, cell_volumes, contour_si=0.01,
                                       near_xy=b["true_xy"], search_radius_m=search_radius_m)
        zband = depth_band(cc, model, b["true_xy"], search_radius_m)
        row = dict(id=b["id"], shape=b["shape"], true_susceptibility_si=b["susceptibility_si"],
                  true_xy_approx=b["true_xy"], centroid_uncertainty_m=b["centroid_uncertainty_m"],
                  centroid_source=b["centroid_source"])
        if blob is None:
            row["recovered"] = False
        else:
            row["recovered"] = True
            # Sign-agnostic peak (round 1's original metric, kept for
            # comparability) -- can be an opposite-sign artifact, not the
            # body itself, once one is present and larger (see
            # recovered_blob_near's docstring). same_sign_* below is the
            # metric that actually answers "how well was this body's own
            # contrast recovered."
            row["peak_susceptibility_si"] = blob["peak_value"]
            row["peak_recovery_ratio"] = (blob["peak_value"] / b["susceptibility_si"]
                                          if b["susceptibility_si"] else None)
            if same_sign_blob is not None:
                row["same_sign_peak_susceptibility_si"] = same_sign_blob["peak_value"]
                row["same_sign_peak_recovery_ratio"] = (
                    same_sign_blob["peak_value"] / b["susceptibility_si"]
                    if b["susceptibility_si"] else None)
                row["sign_flip"] = abs(blob["peak_value"] - same_sign_blob["peak_value"]) > 1e-12
            else:
                row["same_sign_peak_susceptibility_si"] = None
                row["same_sign_peak_recovery_ratio"] = None
                row["sign_flip"] = None
            wx, wy = blob["weighted_centroid_xy"]
            tx, ty = b["true_xy"]
            row["centroid_offset_m"] = float(np.hypot(wx - tx, wy - ty))
            row["recovered_volume_m3_at_0.01SI"] = vol_m3
            row["recovered_volume_n_cells"] = n_vox
            row["depth_band_z_shallow_to_deep_m"] = zband
        rows.append(row)
    return rows
