"""Traditional smooth (L2 Tikhonov) vs newer sparse/compact (IRLS) inversion
for gravity or magnetics -- structurally the same two functions as
mt_inversion/src/invert.py, generalised over the simulation so gravity and
magnetics can share this code. The beta-cooling and alpha_s lessons learned
there (a too-aggressive cooling schedule let poorly-resolved cells drift to
unphysical values once the target misfit turned out unreachable) are carried
over here directly rather than rediscovering them from scratch.
"""
from dataclasses import dataclass, field

import numpy as np
from simpeg import data as simpeg_data
from simpeg import maps
from simpeg import data_misfit, directives, inverse_problem, inversion, optimization, regularization


@dataclass
class InversionResult:
    method: str
    model: np.ndarray
    predicted: np.ndarray
    phi_d: float
    phi_d_target: float
    beta_history: list = field(default_factory=list)
    phi_d_history: list = field(default_factory=list)
    phi_m_history: list = field(default_factory=list)
    phi_history: list = field(default_factory=list)


@dataclass
class JointInversionResult:
    rho: np.ndarray             # density contrast, g/cc
    chi: np.ndarray             # susceptibility contrast, SI
    predicted_grav: np.ndarray
    predicted_mag: np.ndarray
    phi_d: float
    phi_d_target: float
    beta_history: list = field(default_factory=list)
    phi_d_history: list = field(default_factory=list)
    phi_m_history: list = field(default_factory=list)


def _misfit(survey, sim, dobs, std):
    d = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    return data_misfit.L2DataMisfit(data=d, simulation=sim)


# InexactGaussNewton/Minimize's own stopping tolerances (tolF/tolX/tolG)
# default to 0.1 -- loose enough that the optimizer can declare "converged"
# (small step in model space) well before beta has cooled enough to reach
# the target misfit, cutting the run short at whatever iteration count.
# Tightened here so beta keeps cooling until max_iter or the target misfit
# actually stops it, rather than a premature small-step stop.
_TIGHT_TOL = dict(tolF=1e-6, tolX=1e-6, tolG=1e-6)


def run_smooth(survey, sim, dobs, std, mesh, active_cells, m0, mref, chi_target=1.0,
               max_iter=30, alpha_s=0.05, length_scale_x=1.0):
    # length_scale_x, not a fixed alpha_x: WeightedLeastSquares's own docs
    # give alpha_x = (length_scale_x * base_length)**2, where base_length is
    # the mesh's smallest cell edge -- i.e. the smoothness weight is meant to
    # scale with cell size (finer mesh -> smaller cells -> larger alpha_x)
    # to keep the same *physical* smoothing strength as the mesh changes.
    # This project used to pass a bare alpha_x=1.0 across three mesh sizes
    # (350/220/200 m core cells) -- found, not guessed, while diagnosing
    # striped (not blocky) recovered anomalies at 200 m: alpha_x=1.0 is
    # ~40,000x smaller than SimPEG's own length_scale_x=1.0 default at
    # 200 m cells (base_length^2 = 200^2 = 40,000), so smoothing was
    # drastically under-weighted throughout, worse (in absolute-alpha terms)
    # the finer the mesh got -- letting SimPEG compute alpha_x properly
    # fixes that instead of hand-tuning a number that would need
    # re-deriving every time CORE_CELL_M changes.
    dmis = _misfit(survey, sim, dobs, std)
    reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells,
                                              alpha_s=alpha_s,
                                              length_scale_x=length_scale_x,
                                              length_scale_y=length_scale_x,
                                              length_scale_z=length_scale_x,
                                              reference_model=mref)
    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                          **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    save = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list = [
        directives.UpdateSensitivityWeights(),   # depth weighting
        directives.BetaEstimate_ByEig(beta0_ratio=1.0, random_seed=42),
        directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
        directives.TargetMisfit(chifact=chi_target),
        save,
    ]
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)
    pred = sim.dpred(m_rec)
    n_data = dobs.size
    return InversionResult("smooth_L2", m_rec, pred, float(inv_prob.phi_d), chi_target * n_data,
                           beta_history=list(save.beta), phi_d_history=list(save.phi_d),
                           phi_m_history=list(save.phi_m), phi_history=list(save.phi))


def run_sparse(survey, sim, dobs, std, mesh, active_cells, m0, mref, chi_target=1.0,
              norms=(0.0, 0.0, 0.0, 0.0), max_iter=22, alpha_s=0.05, length_scale_x=1.0,
              chifact_start=3.0, max_irls_iterations=8, irls_cooling_factor=1.2):
    # length_scale_x, not alpha_x -- see run_smooth's comment for why.
    dmis = _misfit(survey, sim, dobs, std)
    reg = regularization.Sparse(mesh, active_cells=active_cells,
                                alpha_s=alpha_s,
                                length_scale_x=length_scale_x,
                                length_scale_y=length_scale_x,
                                length_scale_z=length_scale_x,
                                reference_model=mref, norms=list(norms))
    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                          **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    save = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list = [
        directives.UpdateSensitivityWeights(),
        directives.BetaEstimate_ByEig(beta0_ratio=1.0, random_seed=42),
        directives.TargetMisfit(chifact=chi_target),
        # chifact_start > 1: see mt_inversion/README.md -- with the default
        # (1.0), IRLS never engages at all if chi=1 turns out unreachable.
        # irls_cooling_factor (SimPEG default 1.2, how hard beta swings per
        # IRLS step) exposed after finding a real oscillating-divergence
        # case (India/Mayurbhanj magnetics: phi_d undershot the target then
        # bounced back up over the last few IRLS iterations instead of
        # settling -- checked directly on the saved phi_d_history, not
        # assumed slow convergence) -- a gentler value damps that.
        directives.UpdateIRLS(f_min_change=1e-4, max_irls_iterations=max_irls_iterations,
                              chifact_start=chifact_start, irls_cooling_factor=irls_cooling_factor),
        save,
    ]
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)
    pred = sim.dpred(m_rec)
    n_data = dobs.size
    return InversionResult("sparse_IRLS", m_rec, pred, float(inv_prob.phi_d), chi_target * n_data,
                           beta_history=list(save.beta), phi_d_history=list(save.phi_d),
                           phi_m_history=list(save.phi_m), phi_history=list(save.phi))


def run_doi(survey, sim, dobs, std, mesh, active_cells, chi_target, mref1, mref2, **kwargs):
    """Depth-of-investigation (DOI) diagnostic, Oldenburg & Li (1999): invert
    the same data twice from two different *constant* reference models
    (mref1, mref2), starting each run at its own reference (m0 = mref), then
    compare the two recovered models:

        doi(x) = |m1(x) - m2(x)| / |mref1 - mref2|

    Where doi is near 0, both runs land on the same value regardless of
    which reference they started from -- the data is what's pinning that
    cell down (data-driven). Where doi is near 1, each run stayed close to
    its own reference -- the data has no real leverage there and the
    regularization/reference choice is what's determining the value
    (regularization-driven), which is exactly the "how much of the model is
    the data vs the regularization" question this pipeline hasn't answered
    until now. Clipped to [0, 1]: the theoretical bound, occasionally
    exceeded by a few percent from nonlinearity/noise -- values e.g. 1.02
    are not "worse than fully unresolved", they're the same "fully
    unresolved" plus numerical slop.

    Runs `run_smooth` twice (doubles wall time vs a single inversion) --
    everything in `**kwargs` (alpha_s, length_scale_x, max_iter, ...) is
    passed through unchanged to both, so the two runs differ *only* in
    reference model, which is what isolates the reference's effect from
    everything else.
    """
    n_active = int(active_cells.sum())
    m0_1 = np.full(n_active, mref1)
    m0_2 = np.full(n_active, mref2)
    res1 = run_smooth(survey, sim, dobs, std, mesh, active_cells, m0_1, m0_1, chi_target, **kwargs)
    res2 = run_smooth(survey, sim, dobs, std, mesh, active_cells, m0_2, m0_2, chi_target, **kwargs)
    doi = np.abs(res1.model - res2.model) / np.abs(mref1 - mref2)
    doi = np.clip(doi, 0.0, 1.0)
    return res1, res2, doi


def run_joint(survey_grav, sim_grav, dobs_grav, std_grav, survey_mag, sim_mag, dobs_mag, std_mag,
             mesh, active_cells, chi_target=1.0, cross_gradient_beta=100.0, max_iter=20,
             alpha_s=0.05, length_scale_x=1.0, grav_weight=1.0):
    """Cross-gradient joint inversion (Haber & Gazit 2013 formulation, as
    implemented by SimPEG's `regularization.CrossGradient`): invert gravity
    and magnetics *together*, adding a coupling term that penalises the two
    recovered models (density, susceptibility) having structure -- edges/
    gradients -- in different places, without forcing their *values* to
    match (a dense body and a magnetic body need not have the same
    physical-property contrast, just plausibly the same boundaries). This
    is the "use two independently weak, physically different datasets to
    mutually constrain each other" lever -- unlike denser receivers
    (confirmed *not* to help; see README), this uses information already
    in hand more fully, not more of it.

    `sim_grav`/`sim_mag` must be built (forward.build_*_simulation) against
    the *same* `mesh`/`active_cells` passed here -- cross-gradient compares
    gradients cell-by-cell on one shared discretisation, so the two fields'
    receivers can differ (as they generally do) but the model space can't.
    Their `rhoMap`/`chiMap` are overwritten here (via `maps.Wires`) to
    project from one combined [density; susceptibility] model vector --
    whatever map they were built with is discarded, not reused.

    Smooth L2 only (no sparse/IRLS joint variant implemented) -- kept
    deliberately simple for a first joint-inversion pass; cross-gradient
    coupling already adds one nonlinear term, compounding it with IRLS's
    own nonlinearity (and IRLS's own amplitude-runaway tendency, see
    README "Vertical mesh resolution") is a separate thing to get right,
    not assumed to just work by combining both.

    `cross_gradient_beta`: fixed weight on the cross-gradient term relative
    to the two properties' own smoothness regularization -- not
    beta-cooled or otherwise adapted during the run (unlike the main
    data-misfit beta, which is). Picked once per call, not swept here;
    too small and the two models barely influence each other (no better
    than running them independently), too large and cross-gradient can
    dominate the data misfit itself and force spurious shared structure.

    Target misfit is checked on the *combined* (summed) data misfit, not
    each field's own target independently (`directives.MultiTargetMisfits`
    -- the natural per-field alternative -- errors against a regularization
    that includes a `CrossGradient` term; not pursued further for this
    first pass, see comment at the call site in invert.py's git history/
    scratch test if revisiting).
    """
    n_active = int(active_cells.sum())
    wires = maps.Wires(("density", n_active), ("susceptibility", n_active))

    sim_grav.rhoMap = wires.density
    sim_grav.model = None
    sim_mag.chiMap = wires.susceptibility
    sim_mag.model = None

    dmis_g = _misfit(survey_grav, sim_grav, dobs_grav, std_grav)
    dmis_m = _misfit(survey_mag, sim_mag, dobs_mag, std_mag)
    # Both terms are already chi-square-like (L2DataMisfit standardises by
    # std internally), so summing unweighted looks like it should already
    # balance them -- checked directly, it doesn't: unweighted, magnetics
    # dominated the combined gradient and gravity was left at chi~7 while
    # magnetics reached chi~0.7; SimPEG's ScalingMultipleDataMisfits_ByEig
    # (eigenvalue-based auto-balancing) over-corrected the other direction
    # (gravity chi~0.9, magnetics chi~33). grav_weight is a manual,
    # empirically-found balance (not a physical constant) -- see
    # run_joint.py callers for the value used per district.
    dmis = grav_weight * dmis_g + dmis_m

    m0 = np.zeros(2 * n_active)
    reg_g = regularization.WeightedLeastSquares(mesh, active_cells=active_cells, mapping=wires.density,
                                                alpha_s=alpha_s, length_scale_x=length_scale_x,
                                                length_scale_y=length_scale_x, length_scale_z=length_scale_x,
                                                reference_model=m0)
    reg_m = regularization.WeightedLeastSquares(mesh, active_cells=active_cells,
                                                mapping=wires.susceptibility,
                                                alpha_s=alpha_s, length_scale_x=length_scale_x,
                                                length_scale_y=length_scale_x, length_scale_z=length_scale_x,
                                                reference_model=m0)
    reg_x = regularization.CrossGradient(mesh, wire_map=wires, active_cells=active_cells)
    reg = reg_g + reg_m + cross_gradient_beta * reg_x

    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                          **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    save = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list = [
        directives.UpdateSensitivityWeights(),
        directives.BetaEstimate_ByEig(beta0_ratio=1.0, random_seed=42),
        directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
        directives.TargetMisfit(chifact=chi_target),
        save,
    ]
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)

    rho = wires.density * m_rec
    chi = wires.susceptibility * m_rec
    pred_g = sim_grav.dpred(m_rec)
    pred_m = sim_mag.dpred(m_rec)
    n_data = dobs_grav.size + dobs_mag.size
    return JointInversionResult(rho, chi, pred_g, pred_m, float(inv_prob.phi_d),
                                chi_target * n_data, beta_history=list(save.beta),
                                phi_d_history=list(save.phi_d), phi_m_history=list(save.phi_m))
