"""Plotting helpers vendored from grav_mag_inversion/scripts/run_gravity.py
(see VENDORED.md) -- style, plot_convergence, plot_data_fit,
plot_model_sections, _slice_locs, and the shared colour constants only
(the subset scripts/plot_figures.py actually uses). The rest of
run_gravity.py is a driver script coupled to that project's own config.py
and io_data.py, which this repo has no reason to pull in.
"""
import matplotlib.pyplot as plt
import numpy as np

C_S1, C_S2, C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS, C_SURF = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")


def style(ax, title="", subtitle="", xlabel="", ylabel=""):
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=C_INK, fontsize=10.5, fontweight="600", loc="left",
                     pad=14 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=C_INK2, fontsize=8, va="bottom")
    ax.set_xlabel(xlabel, color=C_INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=C_INK2, fontsize=8.5)


def plot_convergence(res, chi_target, n_data, out_path, field_label):
    """phi_d (data misfit) and phi_m (model-norm term) per iteration, on log
    axes, with the target misfit marked -- lets you see whether a run
    actually reached the target or was cut off (by max_iter or an early
    optimizer stop) before getting there."""
    iters = np.arange(1, len(res.phi_d_history) + 1)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(iters, res.phi_d_history, "o-", color=C_S1, ms=4, lw=1.4, label="phi_d (data misfit)")
    ax1.axhline(chi_target * n_data, color=C_MUTED, ls="--", lw=1,
               label=f"target (chi = {chi_target:g})")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    if res.phi_m_history:
        ax2.plot(iters, res.phi_m_history, "s-", color=C_S2, ms=3.5, lw=1.2, alpha=0.85,
                label="phi_m (model norm)")
        ax2.set_yscale("log")
        ax2.tick_params(colors=C_MUTED, labelsize=8)
        ax2.set_ylabel("phi_m", color=C_S2, fontsize=8.5)
    style(ax1, f"{field_label}: convergence", f"{res.method}, per outer iteration",
          xlabel="iteration", ylabel="phi_d")
    ax1.set_ylabel("phi_d", color=C_S1, fontsize=8.5)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = (ax2.get_legend_handles_labels() if res.phi_m_history else ([], []))
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=7.5, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


_DATA_FIT_TITLES = ("Raw observed", "Observed (minus regional trend)",
                    "Predicted, sparse IRLS", "Fit residual (obs - pred)")


def _data_fit_row(axes_row, x, y, raw, dobs, predicted, unit_label, show_titles):
    """One field's 4 data-fit panels, drawn into an existing row of axes.
    Panels 2-4 deliberately share one colour scale, set from the *signal*
    amplitude (dobs/predicted), not from the residual's own (much
    smaller) range -- a residual panel autoscaled to its own max always
    looks fully saturated regardless of how small the true misfit is.
    Sharing the signal's scale instead means the residual panel shows
    mostly background colour, with visible colour only where the misfit
    is actually large relative to the signal."""
    resid = dobs - predicted
    rms = float(np.sqrt(np.mean(resid ** 2)))
    lim = max(np.abs(dobs).max(), np.abs(predicted).max())
    raw_lo, raw_hi = raw.min(), raw.max()

    for ax, dat, title, vmin, vmax in zip(
            axes_row, (raw, dobs, predicted, resid),
            _DATA_FIT_TITLES, (raw_lo, -lim, -lim, -lim), (raw_hi, lim, lim, lim)):
        sc = ax.scatter(x, y, c=dat, cmap="RdBu_r", s=20, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        style(ax, title if show_titles else "", xlabel="easting (m)",
              ylabel="northing (m)" if ax is axes_row[0] else "")
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label=unit_label)
    return rms, lim


def plot_data_fit(x, y, raw, dobs, predicted, unit_label, out_path, field_label):
    """Single-field data-fit figure -- see _data_fit_row."""
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5))
    rms, lim = _data_fit_row(axes, x, y, raw, dobs, predicted, unit_label, show_titles=True)
    fig.suptitle(f"{field_label}: data fit -- panels 2-4 share one colour scale "
                f"(RMS fit residual = {rms:.2f} {unit_label}, {100*rms/lim:.1f}% of that scale)",
                x=0.01, ha="left", color=C_INK, fontsize=11, fontweight="600")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    return rms


def plot_model_sections(mesh, active_idx, model, x_all, y_all, slice_locs, unit_label,
                        out_path, model_title, normal="Y", label_start="A",
                        cmap="RdBu_r", vmin=None, vmax=None, cbar_note="99.5th pctile",
                        depth_lim=None, boundary_loops=None):
    """Several parallel model cross-sections, plus a small plan-view locator
    panel marking where each line sits in the AOI.

    normal="Y" (default): sections run east-west, at fixed northing values
    (`slice_locs` are northing/y coordinates). normal="X": sections run
    north-south, at fixed easting values.

    `label_start` sets the first section letter (e.g. "D" so a north-south
    set doesn't reuse "A"/"B"/"C" from an east-west set already labelled in
    the same report).

    `cmap`/`vmin`/`vmax`/`cbar_note`: default is the diverging density/
    susceptibility rendering (RdBu_r, symmetric range clipped at the
    model's own 99.5th percentile).

    `depth_lim`: (bottom, top) elevation in metres to crop the vertical axis
    to. Default (None) shows the mesh's full modelled depth, including
    coarse padding cells far below the core.

    `boundary_loops`: optional list of (N, 2) arrays (one per closed loop)
    tracing a real survey footprint -- drawn on the locator panel only.
    None (default) skips this.

    Cross-axis (easting or northing) limits are set from the *receiver*
    extent (`x_all`/`y_all`) plus a fixed margin, not the mesh's full
    active-cell extent -- deliberately excludes the coarse octree padding
    beyond the receivers.
    """
    full = np.full(mesh.n_cells, np.nan)
    full[active_idx] = model
    if vmin is None or vmax is None:
        lim = np.nanpercentile(np.abs(model), 99.5)
        vmin, vmax = -lim, lim
    labels = [chr(ord(label_start) + i) for i in range(len(slice_locs))]

    extent_margin = 2000.0
    if normal == "Y":
        cross_idx, cross_label, loc_label = 0, "easting (m)", "y"
        cross_lo, cross_hi = x_all.min() - extent_margin, x_all.max() + extent_margin
    else:
        cross_idx, cross_label, loc_label = 1, "northing (m)", "x"
        cross_lo, cross_hi = y_all.min() - extent_margin, y_all.max() + extent_margin

    n = len(slice_locs)
    fig, axes_all = plt.subplots(1, n + 1, figsize=(4.6 * n + 3.2, 5.5),
                                 gridspec_kw=dict(width_ratios=[1] * n + [0.75], wspace=0.7))
    axes, ax_map = axes_all[:n], axes_all[n]

    for i, (ax, loc, lab) in enumerate(zip(axes, slice_locs, labels)):
        out = mesh.plot_slice(full, normal=normal, slice_loc=loc, ax=ax,
                              pcolor_opts=dict(cmap=cmap, vmin=vmin, vmax=vmax))
        ax.set_title("")   # plot_slice's own auto title collides with style()'s
        ax.set_xlim(cross_lo, cross_hi)
        if depth_lim:
            ax.set_ylim(*depth_lim)
        else:
            ax.set_ylim(mesh.cell_centers[active_idx][:, 2].min() - 500, 500)
        style(ax, f"{lab}-{lab}'", f"{loc_label} = {loc/1000:.1f} km",
              xlabel=cross_label, ylabel="elevation (m)" if i == 0 else "")
        fig.colorbar(out[0], ax=ax, fraction=0.046, pad=0.03,
                    label=f"{unit_label} ({cbar_note})" if cbar_note else unit_label)

    # locator: plan view of the receivers with each section line marked --
    # horizontal lines for east-west sections (normal="Y"), vertical lines
    # for north-south sections (normal="X").
    ax_map.scatter(x_all, y_all, s=1.5, c=C_GRID, linewidths=0)
    if boundary_loops:
        for loop in boundary_loops:
            ax_map.plot(loop[:, 0], loop[:, 1], color=C_S2, lw=1.1, alpha=0.85)
    ax_map.set_aspect("equal")
    x0, x1 = x_all.min(), x_all.max()
    y0, y1 = y_all.min(), y_all.max()
    pad_x = 0.08 * (x1 - x0)
    pad_y = 0.08 * (y1 - y0)
    for loc, lab in zip(slice_locs, labels):
        if normal == "Y":
            ax_map.axhline(loc, color=C_S1, lw=1.4)
            ax_map.text(x0 - pad_x, loc, lab, va="center", ha="right", fontsize=8,
                       color=C_INK, fontweight="600", clip_on=False)
            ax_map.text(x1 + pad_x, loc, f"{lab}'", va="center", ha="left", fontsize=8,
                       color=C_INK, fontweight="600", clip_on=False)
        else:
            ax_map.axvline(loc, color=C_S1, lw=1.4)
            ax_map.text(loc, y1 + pad_y, lab, va="bottom", ha="center", fontsize=8,
                       color=C_INK, fontweight="600", clip_on=False)
            ax_map.text(loc, y0 - pad_y, f"{lab}'", va="top", ha="center", fontsize=8,
                       color=C_INK, fontweight="600", clip_on=False)
    ax_map.set_xlim(x0 - pad_x, x1 + pad_x)
    ax_map.set_ylim(y0 - pad_y, y1 + pad_y)
    ax_map.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax_map.tick_params(axis="x", labelrotation=30)
    style(ax_map, "Section\nlocations", "plan view", xlabel="easting (m)", ylabel="northing (m)")
    ax_map.yaxis.set_label_position("right")
    ax_map.yaxis.tick_right()
    fig.suptitle(model_title, x=0.01, ha="left", color=C_INK, fontsize=11.5, fontweight="600")
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.045, right=0.965)
    plt.savefig(out_path, dpi=150)


def _slice_locs(lo, hi):
    """Three evenly-spaced section positions spanning [lo, hi] (20/50/80%
    of the way across)."""
    return [lo + f * (hi - lo) for f in (0.2, 0.5, 0.8)]
