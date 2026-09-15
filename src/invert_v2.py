"""Round-2 inversion runner -- structurally IDENTICAL to grav_mag_inversion.
src.invert.run_smooth (same regularization.WeightedLeastSquares, same
optimization.InexactGaussNewton, same directive list/tolerances), with two
additive toggles wired in locally. Deliberately a separate function in
THIS project rather than an edit to the shared grav_mag_inversion/src/
invert.py -- that file is also used by the WA and India projects, and the
task's constraint ("current Yilgarn workflow must still run unchanged")
is met most directly by not touching it at all, rather than by adding
optional/default-off parameters to it. Neither
`regularization.WeightedLeastSquares` nor `optimization.
InexactGaussNewton` is subclassed or monkeypatched -- both toggles use
only their existing public APIs (`reg.set_weights()`, and simply not
adding the `UpdateSensitivityWeights` directive when distance weighting
is requested).

Toggles:
  - `weighting`: "depth" (default -- reproduces run_smooth exactly, via
    directives.UpdateSensitivityWeights) or "distance" (src/
    distance_weighting.py's Li & Oldenburg 1996b weights, injected via
    reg.set_weights(sensitivity=...) -- the same weight key
    UpdateSensitivityWeights itself writes to, confirmed by reading its
    source, so this is a genuine substitution of one weighting scheme
    for the other, not an additional independent term).
  - `cache_key`: None (default -- no caching, matches run_smooth) or a
    string from src/kernel_cache.cache_key(...) to point the simulation
    at a hash-keyed disk cache before the first G access.
  - `beta_floor`: None (default -- lets BetaSchedule cool beta as low as
    TargetMisfit needs, matching run_smooth) or a float, clamping beta
    from cooling below it. Added after empirically finding that once
    real topography is on, beta cools 500x+ further than the flat-
    topography baseline needed regardless of WHICH sensitivity weighting
    scheme is used (distance, fitted-r0 distance, or real per-cell
    sensitivity all show the same pattern) -- this cools straight through
    the level of model complexity that keeps every body's recovery
    correctly signed. `beta_floor` trades a worse final chi for staying
    above that level; see improved_report.md's beta-floor section for
    what floor value that turned out to be, checked directly against
    these runs' own beta histories, not guessed.
  - `bounds`: None (default -- plain InexactGaussNewton, matches
    run_smooth) or a (lower, upper) tuple, switching the optimizer to
    `optimization.ProjectedGNCG`. Added per the paper's OWN actual
    regularization choice for these benchmarks (checked directly from the
    article text, Sect. on inversion setup): "The positivity constraint
    was added via the alternating direction method of multipliers
    (ADMM)-bound constraints... In all susceptibility inversions a
    positivity constraint was applied to reduce the model non-uniqueness
    (Li and Oldenburg, 1996a)." -- i.e. Ogarko et al. do NOT rely on a
    weighting scheme or beta schedule to suppress negative artifacts;
    they hard-constrain susceptibility to a physical range (0 to 1 SI in
    their TOMOPARAMS.TXT, `inversion.admm.magn.bounds`). ProjectedGNCG is
    used here instead of their ADMM implementation specifically (ADMM
    itself is a different, more general constrained-optimisation
    framework -- ProjectedGNCG is SimPEG's own bound-constraint mechanism,
    confirmed by reading its source to inherit from `InexactGaussNewton`
    with a `Bounded` mixin added -- i.e. the same Gauss-Newton steps,
    just projected back into [lower, upper] each step, not a different
    regularization or optimizer algorithm)."""
import sys
from pathlib import Path

import numpy as np
from simpeg import data_misfit, directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg import data as simpeg_data


class _BetaFloor(directives.InversionDirective):
    """Clamps inv_prob.beta from cooling below `floor` -- BetaSchedule
    divides beta each iteration with no lower bound of its own (checked
    directly from its source: no min-beta parameter exists), so this
    directive must run AFTER it in the directive list to have effect."""
    def __init__(self, floor, **kwargs):
        super().__init__(**kwargs)
        self.floor = floor

    def endIter(self):
        if self.invProb.beta < self.floor:
            self.invProb.beta = self.floor


class _MClip(directives.InversionDirective):
    """Clamps the (log-space) model to [lo, hi] after each step -- a
    safety net for run_positivity_v2's chi=exp(m) reparameterization,
    added after the first attempt overflowed: with beta0_ratio=1.0 (the
    same value run_smooth_v2/run_sparse_v2 use, both linear-model
    problems), the very first Gauss-Newton step proposed an m large
    enough that exp(m) overflowed float32 (confirmed from the traceback:
    "overflow encountered in cast"/"in matmul", SimPEG's dpred() then
    correctly refusing to proceed on a NaN/inf-contaminated prediction
    rather than silently continuing). Default bounds [-25, 5] correspond
    to chi in [~1.4e-11, ~148] SI -- generous relative to every value any
    configuration this round has recovered (largest seen: ~0.09 SI), so
    this is a numerical safety rail, not a physical bound doing the
    positivity's job (chi=exp(m) already guarantees chi>0 on its own,
    with or without this clip)."""
    def __init__(self, lo=-25.0, hi=5.0, **kwargs):
        super().__init__(**kwargs)
        self.lo, self.hi = lo, hi

    def endIter(self):
        np.clip(self.invProb.model, self.lo, self.hi, out=self.invProb.model)


class _PositivityClip(directives.InversionDirective):
    """Post-hoc heuristic positivity: clamps the LINEAR susceptibility
    model to [lo, hi] after each step, on a plain (unconstrained)
    InexactGaussNewton run -- NOT a real bound-constrained optimization
    (the line search inside that step still saw an unclipped model; this
    only rewrites `invProb.model` afterwards, the same public hook
    `_BetaFloor`/`_MClip` already use, so it touches nothing in
    `optimization.InexactGaussNewton`/`regularization.Sparse` themselves).

    Tried as the fast/cheap alternative to fixing `optimization.
    ProjectedGNCG`'s two failed numerical attempts (see improved_report.
    md's "Root cause investigation") -- this either kills the sign-flip
    artifact outright (evidence the artifact really is just "the
    optimizer wants to go negative and nothing stops it", not something
    structural to the data/mesh) or it doesn't (leaving the artifact
    fully intact even under a hard floor would be equally informative --
    it would mean the negative structure re-emerges through some other
    mechanism, e.g. the reference model or the sparsity weights
    themselves pulling cells back down every iteration after the clip).

    Default hi=1.0 SI matches the paper's own TOMOPARAMS.TXT admm.magn.
    bounds (0 to 1 SI) -- not because any run has come close to it (the
    largest susceptibility seen in ANY configuration so far is ~0.09 SI,
    see _MClip's docstring), just for exact fidelity to their own actual
    constraint rather than picking a new number."""
    def __init__(self, lo=0.0, hi=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lo, self.hi = lo, hi

    def endIter(self):
        np.clip(self.invProb.model, self.lo, self.hi, out=self.invProb.model)

class _BoundsDiagnostic(directives.InversionDirective):
    """Logs beta, phi_d, phi_m, and how many active cells sit AT the
    bound (within 1e-9 of lo or hi) after every iteration -- added
    because both earlier `bounds=` attempts (see improved_report.md's
    "Root cause investigation") were debugged from just the printed
    SimPEG convergence table and a traceback, then abandoned rather than
    inspected further. This answers, directly, the two questions that
    table can't: is the model actually moving (vs. stuck exactly at
    m0), and is the active-set (cells pinned at a bound) exploding right
    before a line-search failure -- rather than guessing from symptoms."""
    def __init__(self, lo, hi, **kwargs):
        super().__init__(**kwargs)
        self.lo, self.hi = lo, hi

    def endIter(self):
        m = self.invProb.model
        n_lo = int(np.sum(np.abs(m - self.lo) < 1e-9)) if self.lo is not None else 0
        n_hi = int(np.sum(np.abs(m - self.hi) < 1e-9)) if self.hi is not None else 0
        print(f"    [bounds diag] beta={self.invProb.beta:.4g}  "
             f"model[min={m.min():.5g}, max={m.max():.5g}]  "
             f"at_lo={n_lo}  at_hi={n_hi}  n_active={m.size}")


_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
from vendor.grav_mag_inversion.invert import InversionResult, _TIGHT_TOL   # noqa: E402
from src import distance_weighting, kernel_cache                           # noqa: E402


def run_smooth_v2(survey, sim, dobs, std, mesh, active_cells, m0, mref, chi_target=1.0,
                  max_iter=30, alpha_s=0.05, length_scale_x=1.0,
                  weighting="depth", distance_weighting_beta=3.0, distance_weighting_r0=None,
                  core_cell_z_m=None, cache_key=None, cache_dir=kernel_cache.DEFAULT_CACHE_DIR,
                  beta_floor=None, bounds=None, beta0_ratio=1.0, use_preconditioner=False,
                  bounds_diagnostic=False, cooling_factor=2.0, cooling_rate=1):
    """`bounds=(lo, hi)` (see the class docstring at the top of this
    module for why `ProjectedGNCG`, not a different optimizer/regulariser)
    failed twice before with default `beta0_ratio=1.0` and no
    preconditioner (see improved_report.md's "Root cause investigation"):
    `m0` exactly at the lower bound got stuck (the projected step likely
    collapsed to zero at the boundary); `m0` just inside it broke the
    line search at iteration 7. Two changes, both borrowed from the
    ONE bound-avoiding fix that did work elsewhere in this module
    (`run_positivity_v2`'s chi=exp(m) needed `beta0_ratio=100` to stop
    its first step overflowing) -- `beta0_ratio` (try higher than 1.0
    here too, so the first step is more conservative, less likely to
    demand a jump the projection can't accommodate) and
    `use_preconditioner` (SimPEG's own repeated warning on every
    topography run so far, never actually tried on the bounds= path).
    `bounds_diagnostic=True` appends `_BoundsDiagnostic` so a THIRD
    failure is at least explained by per-iteration beta/active-set
    numbers, not just a traceback -- neither previous attempt did this.

    `cooling_factor`/`cooling_rate` (default 2.0/1 -- SimPEG's own
    default, unchanged from the two failed attempts and from
    run_smooth_v2's own non-bounded path): once `bounds=` actually got a
    30/60-iteration run to complete without crashing (this round's first
    real success), it stalled at chi=1.38 with beta cooled down to
    ~1e-14 after 60 iterations -- i.e. racing beta down every single
    iteration outpaced how fast the projected active-set solver could
    actually settle at each level, so TargetMisfit never got a chance to
    fire before beta was already negligible and iterations ran out.
    Slower cooling (larger cooling_rate = more GN iterations held at
    each beta value; smaller cooling_factor = gentler steps down) gives
    the active set time to stabilise before the next drop -- untested
    with the default schedule inherited by accident rather than chosen
    for this optimizer."""
    if weighting not in ("depth", "distance"):
        raise ValueError(f"weighting must be 'depth' or 'distance', got {weighting!r}")

    if cache_key is not None:
        sim, _path, is_new = kernel_cache.attach_cache(sim, cache_key, cache_dir)
    else:
        is_new = None

    d = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=d, simulation=sim)
    reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells,
                                              alpha_s=alpha_s,
                                              length_scale_x=length_scale_x,
                                              length_scale_y=length_scale_x,
                                              length_scale_z=length_scale_x,
                                              reference_model=mref)

    directive_list = [directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=42),
                      directives.BetaSchedule(coolingFactor=cooling_factor, coolingRate=cooling_rate),
                      directives.TargetMisfit(chifact=chi_target)]
    if beta_floor is not None:
        directive_list.append(_BetaFloor(beta_floor))   # after BetaSchedule -- see class docstring
    if use_preconditioner:
        directive_list.insert(1, directives.UpdatePreconditioner())
    if bounds_diagnostic and bounds is not None:
        directive_list.append(_BoundsDiagnostic(*bounds))

    if weighting == "depth":
        directive_list.insert(0, directives.UpdateSensitivityWeights())
    else:
        receiver_locations = survey.source_field.receiver_list[0].locations
        w = distance_weighting.distance_weights(
            mesh, active_cells, receiver_locations, beta=distance_weighting_beta,
            r0=distance_weighting_r0, core_cell_z_m=core_cell_z_m)
        reg.set_weights(sensitivity=w)

    if bounds is None:
        opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                              **_TIGHT_TOL)
    else:
        lower, upper = bounds
        opt = optimization.ProjectedGNCG(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                         lower=lower, upper=upper, **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    save = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list.append(save)
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)
    pred = sim.dpred(m_rec)
    n_data = dobs.size
    res = InversionResult("smooth_L2_v2", m_rec, pred, float(inv_prob.phi_d), chi_target * n_data,
                          beta_history=list(save.beta), phi_d_history=list(save.phi_d),
                          phi_m_history=list(save.phi_m), phi_history=list(save.phi))
    return res, is_new


def run_sparse_v2(survey, sim, dobs, std, mesh, active_cells, m0, mref, chi_target=1.0,
                  norms=(0.0, 2.0, 2.0, 2.0), max_iter=22, alpha_s=0.05, length_scale_x=1.0,
                  chifact_start=3.0, max_irls_iterations=12, irls_cooling_factor=1.1,
                  cache_key=None, cache_dir=kernel_cache.DEFAULT_CACHE_DIR,
                  positivity_clip=None):
    """Sparse/IRLS -- structurally identical to grav_mag_inversion.src.
    invert.run_sparse (same regularization.Sparse, same
    optimization.InexactGaussNewton -- IRLS uses UpdateIRLS's own beta
    handling, not BetaSchedule, so bounds/beta_floor don't apply here the
    way they do to run_smooth_v2), only adding kernel-cache support. Same
    default norms=(0,2,2,2) and irls_cooling_factor=1.1/
    max_irls_iterations=12 as india_grav_mt_inversion's own run_magnetics.
    py uses (found there to fix an oscillating-convergence case; kept as
    the starting point here rather than SimPEG's own defaults (1.2/8), on
    the reasoning that both projects' data share the same "wide dynamic
    range" symptom that motivated the change there).

    Requested specifically because round 2's investigation (improved_
    report.md's "Root cause investigation" section) found real topography
    needs a much lower beta than smooth L2's flat baseline to fit
    comparably well, and low beta is exactly the regime smooth L2's own
    ringing/side-lobe artefacts show up in -- sparse/IRLS's minimum-
    support norm on alpha_s (norms[0]=0) is what grav_mag_inversion's own
    WA/India projects adopted to fix the same class of problem (see their
    READMEs), not a mechanism tried and rejected this round.

    `positivity_clip`: None (default -- matches the shipped result
    exactly) or (lo, hi) to append `_PositivityClip` after every
    iteration -- the cheap heuristic alternative to fixing
    `optimization.ProjectedGNCG`'s two failed attempts (see
    improved_report.md's "Root cause investigation"); see
    _PositivityClip's own docstring for exactly what this does and does
    not prove."""
    if cache_key is not None:
        sim, _path, is_new = kernel_cache.attach_cache(sim, cache_key, cache_dir)
    else:
        is_new = None

    d = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=d, simulation=sim)
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
        directives.UpdateIRLS(f_min_change=1e-4, max_irls_iterations=max_irls_iterations,
                              chifact_start=chifact_start, irls_cooling_factor=irls_cooling_factor),
        save,
    ]
    if positivity_clip is not None:
        directive_list.append(_PositivityClip(*positivity_clip))
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)
    pred = sim.dpred(m_rec)
    n_data = dobs.size
    method_name = "sparse_IRLS_v2" if positivity_clip is None else "sparse_IRLS_v2_posclip"
    res = InversionResult(method_name, m_rec, pred, float(inv_prob.phi_d), chi_target * n_data,
                          beta_history=list(save.beta), phi_d_history=list(save.phi_d),
                          phi_m_history=list(save.phi_m), phi_history=list(save.phi))
    return res, is_new


def run_positivity_v2(survey, sim, dobs, std, mesh, active_cells, chi_target=1.0,
                      regularization_type="smooth", max_iter=30, alpha_s=0.05, length_scale_x=1.0,
                      m0_value=1e-4, mref_value=1e-4, beta0_ratio=100.0, m_clip=(-25.0, 5.0),
                      cooling_factor=2.0, cooling_rate=2, use_preconditioner=False,
                      norms=(0.0, 2.0, 2.0, 2.0), max_irls_iterations=12, irls_cooling_factor=1.1,
                      chifact_start=3.0, cache_key=None, cache_dir=kernel_cache.DEFAULT_CACHE_DIR):
    """Positivity via reparameterization (chi = exp(m)), NOT via
    `optimization.ProjectedGNCG` bound projection -- the alternative to
    the bounds= path in run_smooth_v2, tried after that one failed
    numerically twice (see improved_report.md's "Root cause
    investigation"). `sim.chiMap` is reassigned from `maps.IdentityMap`
    (what forward.build_magnetic_simulation constructs it with) to
    `maps.ExpMap` -- confirmed directly (not assumed) that this is a
    normal, supported reassignment in SimPEG (chiMap is a public,
    settable attribute; a tiny standalone smoke test reproduced the
    expected `sim.chi = exp(sim.model)` behaviour before this was used
    here). chi = exp(m) is always > 0 by construction, for ANY real m --
    the optimizer runs entirely unconstrained (plain InexactGaussNewton,
    not ProjectedGNCG), so there is no projection step, no active set, and
    none of the numerical issues that broke the bounds= attempts. No
    upper bound is enforced (unlike the paper's 0-1 SI ADMM constraint) --
    checked directly against every run so far: no configuration's peak
    susceptibility got anywhere near 1 SI (largest seen: ~0.09 SI), so the
    upper bound was never the binding constraint; only positivity is.

    IMPORTANT -- everything about m0/mref/reference_model is in LOG space
    here, not linear susceptibility: `m0_value`/`mref_value` are the
    LINEAR starting/reference susceptibility (e.g. 1e-4 SI, a small
    background value), converted to log space internally
    (`np.log(m0_value)`) before being handed to the optimizer/
    regularization -- passing the old m0=0 convention here would mean
    chi=exp(0)=1 SI, i.e. starting at the paper's own upper bound, wildly
    wrong. The regularization is built with `mapping=chiMap` (the same
    ExpMap), so smallness/smoothness penalise the actual LINEAR
    susceptibility field, not its log -- SimPEG's own standard pattern
    for a mapped physical property (confirmed by reading regularization.
    base's smallness term: `mapping*m - mapping*mref`), not something
    invented for this round.

    `beta0_ratio` (default 100, not `run_smooth_v2`/`run_sparse_v2`'s
    1.0) and `m_clip`: both exist because the first attempt at this
    (beta0_ratio=1.0, no clip) crashed on iteration 0 -- confirmed from
    the traceback, not assumed: the auto-estimated starting beta was
    small enough that the first Gauss-Newton step proposed an m large
    enough for exp(m) to overflow float32 ("overflow encountered in
    cast"/"in matmul", then SimPEG's own dpred() correctly refusing to
    proceed on a NaN/inf-contaminated prediction). `beta0_ratio=100`
    makes that first step far more conservative (more regularisation
    pull relative to the data term, a real, disclosed tuning choice, not
    SimPEG's own default). `m_clip` is a secondary safety net for LATER
    iterations (an `endIter` hook can't prevent iteration 0's crash,
    which happens inside a line search before any `endIter` fires --
    beta0_ratio is what actually fixes that one) -- see _MClip's own
    docstring for why [-25, 5] is generous, not a physical bound doing
    positivity's job."""
    if regularization_type not in ("smooth", "sparse"):
        raise ValueError(f"regularization_type must be 'smooth' or 'sparse', got {regularization_type!r}")

    n_active = int(active_cells.sum())
    chi_map = maps.ExpMap(nP=n_active)
    sim.chiMap = chi_map   # reassignment confirmed supported -- see docstring

    if cache_key is not None:
        sim, _path, is_new = kernel_cache.attach_cache(sim, cache_key, cache_dir)
    else:
        is_new = None

    m0 = np.full(n_active, np.log(m0_value))
    mref = np.full(n_active, np.log(mref_value))

    d = simpeg_data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=d, simulation=sim)

    if regularization_type == "smooth":
        reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells,
                                                  mapping=chi_map, alpha_s=alpha_s,
                                                  length_scale_x=length_scale_x,
                                                  length_scale_y=length_scale_x,
                                                  length_scale_z=length_scale_x,
                                                  reference_model=mref)
        # coolingRate=2, not run_smooth_v2's 1: chi=exp(m) makes the
        # problem genuinely nonlinear in m (run_smooth_v2's linear-model
        # case used SimPEG's own coolingRate=1 recommendation "for linear
        # least-squares problems" -- BetaSchedule's docstring -- that no
        # longer applies once the model-to-data map isn't linear).
        directive_list = [directives.UpdateSensitivityWeights(),
                          directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=42),
                          directives.BetaSchedule(coolingFactor=cooling_factor, coolingRate=cooling_rate),
                          directives.TargetMisfit(chifact=chi_target)]
        method_name = "smooth_L2_expmap_v2"
    else:
        reg = regularization.Sparse(mesh, active_cells=active_cells,
                                    mapping=chi_map, alpha_s=alpha_s,
                                    length_scale_x=length_scale_x,
                                    length_scale_y=length_scale_x,
                                    length_scale_z=length_scale_x,
                                    reference_model=mref, norms=list(norms))
        directive_list = [directives.UpdateSensitivityWeights(),
                          directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=42),
                          directives.TargetMisfit(chifact=chi_target),
                          directives.UpdateIRLS(f_min_change=1e-4, max_irls_iterations=max_irls_iterations,
                                                chifact_start=chifact_start,
                                                irls_cooling_factor=irls_cooling_factor)]
        method_name = "sparse_IRLS_expmap_v2"

    if use_preconditioner:
        # SimPEG's own repeated warning on every expmap run so far:
        # "Without a Linear preconditioner, convergence may be slow.
        # Consider adding directives.UpdatePreconditioner" -- never
        # tried until now. A Jacobi preconditioner for the CG sub-solve,
        # not a regularization or optimizer change.
        directive_list.insert(1, directives.UpdatePreconditioner())

    opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
                                          **_TIGHT_TOL)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    save = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list.append(save)
    if m_clip is not None:
        directive_list.append(_MClip(*m_clip))   # after save doesn't matter -- endIter order among these two is irrelevant
    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_rec = inv.run(m0)
    chi_rec = chi_map * m_rec   # back to linear susceptibility for every downstream consumer
    pred = sim.dpred(m_rec)
    n_data = dobs.size
    res = InversionResult(method_name, chi_rec, pred, float(inv_prob.phi_d), chi_target * n_data,
                          beta_history=list(save.beta), phi_d_history=list(save.phi_d),
                          phi_m_history=list(save.phi_m), phi_history=list(save.phi))
    return res, is_new
