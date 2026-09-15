"""Flags the four L2-smooth recovery artefacts Ogarko et al. (2024)
document, checked against each Synthetic400 body's approximate true
geometry (src/validation.py's BODIES -- same digitised-position caveats
apply here: these are heuristic geometric checks against an approximate
truth, not a formal statistical test. Each flag is a bool + a short
evidence string with the actual numbers behind it, so a flag can be
read/disputed rather than taken on faith.

The four artefacts, and which body type each is checked against:
  1. Deep-tail extension below a high-susceptibility body -- checked
     against Body 4 (0.10 SI, the strongest true source; a compact body
     should not need a long tail of significant recovered value below it
     to fit the data).
  2. Positive-susceptibility halo around an inferred negative-contrast
     body -- checked against Body 6 (0.0015 SI, "magnetite destruction"
     analogue, the one body framed as a *contrast deficit* rather than a
     positive source -- see baseline_report.md's Body 6 discussion for
     why round 1 also flagged this qualitatively).
  3. Over-estimated bottom depth of thin sheets -- checked against Bodies
     2, 3, 5 (all "slab" shapes, true thickness 10-50 m per the paper's
     text -- see load_survey.py's module docstring for that
     source's lower confidence level).
  4. Bowl-shaped over-thickening of thin discs -- checked against Body 1
     (the one "disc" shape) -- tests whether the recovered anomaly's
     horizontal footprint grows with depth (a smooth-inversion signature)
     rather than staying roughly disc-like.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import validation as val   # noqa: E402 -- see per_body_report for the same BODIES/search pattern


def _near(cc, xy, radius_m):
    dx = cc[:, 0] - xy[0]
    dy = cc[:, 1] - xy[1]
    return (dx * dx + dy * dy) <= radius_m ** 2


def deep_tail_flag(cc, model, body, search_radius_m=1500.0, tail_depth_m=1200.0,
                   significance_frac=0.3):
    """True if cells more than `tail_depth_m` below the body's shallowest
    significant recovery still carry >= significance_frac of the local
    peak value -- a compact source shouldn't need that. Only meaningful
    for a body expected to be a strong, reasonably localised source
    (Body 4 here)."""
    mask = _near(cc, body["true_xy"], search_radius_m)
    if not mask.any():
        return False, "no active cells found near this body's approximate position"
    vals = np.abs(model[mask])
    z = cc[mask, 2]
    peak = vals.max()
    sig = z[vals >= significance_frac * peak]
    if sig.size == 0:
        return False, "no cells reach the significance threshold"
    shallow, deep = sig.max(), sig.min()
    extent = shallow - deep
    flagged = extent > tail_depth_m
    ev = (f"significant recovery ({significance_frac:.0%} of peak {peak:.4f} SI) spans "
         f"{deep:.0f} to {shallow:.0f} m ({extent:.0f} m total) near Body {body['id']}; "
         f"flag threshold {tail_depth_m:.0f} m")
    return flagged, ev


def negative_halo_flag(cc, model, body, search_radius_m=1200.0, halo_radius_m=(1200.0, 2500.0),
                       pos_threshold_frac=0.5):
    """True if the shell between halo_radius_m[0] and [1] around the
    body's position contains positive values reaching at least
    pos_threshold_frac of the (absolute) core-region peak -- the
    signature of a positive side-lobe flanking a body that should read
    as a contrast deficit, not a real second positive source. Only
    meaningful for a body expected to be a negative/low-contrast source
    (Body 6 here)."""
    core = _near(cc, body["true_xy"], search_radius_m)
    if not core.any():
        return False, "no active cells found in the core region near this body"
    core_peak_abs = np.abs(model[core]).max()

    dx = cc[:, 0] - body["true_xy"][0]
    dy = cc[:, 1] - body["true_xy"][1]
    r = np.hypot(dx, dy)
    shell = (r >= halo_radius_m[0]) & (r <= halo_radius_m[1])
    if not shell.any():
        return False, "no active cells found in the halo shell (mesh/coverage too small)"
    shell_pos_peak = model[shell].max() if (model[shell] > 0).any() else 0.0
    flagged = shell_pos_peak >= pos_threshold_frac * core_peak_abs
    ev = (f"core |peak| near Body {body['id']} = {core_peak_abs:.4f} SI; "
         f"max positive value in the {halo_radius_m[0]:.0f}-{halo_radius_m[1]:.0f} m "
         f"halo shell = {shell_pos_peak:.4f} SI "
         f"({100*shell_pos_peak/core_peak_abs if core_peak_abs else 0:.0f}% of core peak, "
         f"flag threshold {100*pos_threshold_frac:.0f}%)")
    return flagged, ev


def thin_sheet_depth_flag(cc, model, body, search_radius_m=1000.0, significance_frac=0.5,
                          true_thickness_m=50.0, overestimate_ratio=3.0):
    """True if the recovered significant-value vertical extent near a
    thin-sheet body exceeds `overestimate_ratio` times its true (thin)
    thickness. `true_thickness_m` defaults to 50 m (the thickest of the
    slab bodies' reported thicknesses, per module docstring's confidence
    caveat) as a conservative shared default; pass the specific body's
    own value if known more precisely."""
    mask = _near(cc, body["true_xy"], search_radius_m)
    if not mask.any():
        return False, "no active cells found near this body's approximate position"
    vals = np.abs(model[mask])
    z = cc[mask, 2]
    peak = vals.max()
    sig = z[vals >= significance_frac * peak]
    if sig.size == 0:
        return False, "no cells reach the significance threshold"
    extent = sig.max() - sig.min()
    flagged = extent > overestimate_ratio * true_thickness_m
    ev = (f"recovered significant-value extent near Body {body['id']} ({body['shape']}) = "
         f"{extent:.0f} m vs. true thickness ~{true_thickness_m:.0f} m "
         f"({extent/true_thickness_m:.1f}x, flag threshold {overestimate_ratio:.0f}x)")
    return flagged, ev


def bowl_thickening_flag(cc, model, body, search_radius_m=1500.0, z_bins=6,
                         significance_frac=0.4, growth_ratio=1.8):
    """True if the horizontal footprint (count of significant cells at a
    given depth layer) grows by more than `growth_ratio` from the
    shallowest to some deeper bin near a thin-disc body -- a disc should
    keep a roughly constant footprint with depth if recovered faithfully;
    smooth L2 spreading it into a wider, deeper "bowl" is the artefact
    being tested for."""
    mask = _near(cc, body["true_xy"], search_radius_m)
    if not mask.any():
        return False, "no active cells found near this body's approximate position"
    vals = np.abs(model[mask])
    z = cc[mask, 2]
    peak = vals.max()
    sig_mask = vals >= significance_frac * peak
    if sig_mask.sum() < z_bins:
        return False, "too few significant cells to bin by depth"
    zs = z[sig_mask]
    edges = np.linspace(zs.min(), zs.max(), z_bins + 1)
    counts = np.histogram(zs, bins=edges)[0]
    counts = counts[counts > 0]
    if counts.size < 2:
        return False, "significant recovery confined to a single depth bin -- can't assess shape"
    shallow_count, max_count = counts[-1], counts.max()   # zs sorted low->high bin edges; last bin = shallowest
    ratio = max_count / max(shallow_count, 1)
    flagged = ratio > growth_ratio
    ev = (f"significant-cell count near Body {body['id']} grows {ratio:.1f}x from the "
         f"shallowest depth bin ({shallow_count} cells) to the widest bin ({max_count} cells); "
         f"flag threshold {growth_ratio:.1f}x")
    return flagged, ev


def detect_all(cc, model):
    """Runs all four checks against the relevant body/bodies, returns a
    flat list of dicts (artefact, body_id, flagged, evidence) -- one row
    per (artefact type, relevant body) pair."""
    by_id = {b["id"]: b for b in val.BODIES}
    rows = []

    flagged, ev = deep_tail_flag(cc, model, by_id[4])
    rows.append(dict(artefact="deep_tail_extension", body_id=4, flagged=flagged, evidence=ev))

    flagged, ev = negative_halo_flag(cc, model, by_id[6])
    rows.append(dict(artefact="positive_halo_around_negative_body", body_id=6,
                     flagged=flagged, evidence=ev))

    for bid, thickness in ((2, 50.0), (3, 10.0), (5, 20.0)):
        flagged, ev = thin_sheet_depth_flag(cc, model, by_id[bid], true_thickness_m=thickness)
        rows.append(dict(artefact="thin_sheet_bottom_depth_overestimate", body_id=bid,
                         flagged=flagged, evidence=ev))

    flagged, ev = bowl_thickening_flag(cc, model, by_id[1])
    rows.append(dict(artefact="bowl_shaped_disc_overthickening", body_id=1,
                     flagged=flagged, evidence=ev))

    return rows
