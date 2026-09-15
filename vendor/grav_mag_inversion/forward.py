"""Gravity and magnetic forward operators, both verified against SimPEG's
own analytic sphere solutions before use (see tests/).

Conventions established during that verification -- not assumed from memory:
  - Mesh/receiver coordinates: x=easting, y=northing, z=up (confirmed via
    the magnetics sphere test: with declination=0 the field's horizontal
    component landed entirely on the mesh's y-axis, i.e. y=north).
  - Gravity: Simulation3DIntegral density model in g/cc, output in mGal.
  - Magnetics: Simulation3DIntegral susceptibility model in SI (dimensionless),
    output (TMI) in nT, inducing field via UniformBackgroundField(amplitude_nT,
    inclination_deg, declination_deg).
"""
import numpy as np
from simpeg import maps
from simpeg.potential_fields import gravity, magnetics


def build_gravity_simulation(mesh, active_cells, x_m, y_m, z_m=None):
    z_m = np.zeros_like(x_m) if z_m is None else z_m
    locs = np.c_[x_m, y_m, z_m]
    rx = gravity.receivers.Point(locs, components="gz")
    survey = gravity.Survey(gravity.sources.SourceField(receiver_list=[rx]))
    sim = gravity.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(nP=int(active_cells.sum())),
        active_cells=active_cells)
    return survey, sim


def build_magnetic_simulation(mesh, active_cells, x_m, y_m, amplitude_nt, inclination_deg,
                              declination_deg, z_m=None):
    z_m = np.zeros_like(x_m) if z_m is None else z_m
    locs = np.c_[x_m, y_m, z_m]
    rx = magnetics.receivers.Point(locs, components="tmi")
    src = magnetics.UniformBackgroundField(receiver_list=[rx], amplitude=amplitude_nt,
                                           inclination=inclination_deg, declination=declination_deg)
    survey = magnetics.Survey(src)
    sim = magnetics.Simulation3DIntegral(
        mesh, survey=survey, chiMap=maps.IdentityMap(nP=int(active_cells.sum())),
        active_cells=active_cells)
    return survey, sim
