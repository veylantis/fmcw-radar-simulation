"""
Radar target factory for RadarSimPy simulations.

Each factory function returns a RadarTarget dataclass with metadata
and a .to_sim() method that converts it into a RadarSimPy-compatible dict.

Usage:
    from lib.radar_targets import corner_reflector, sphere, point_target

    t = corner_reflector(location=(2, 5, 0), a_leg=0.15, lam=cfg.lam)
    targets_sim = [t.to_sim()]
    print(t.label, t.rcs_dBsm)

Target types:
    corner_reflector()  — trihedral corner reflector
    sphere()            — metallic sphere (optical region)
    point_target()      — arbitrary point target with given RCS
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RadarTarget:
    """
    Point target with metadata.
    .to_sim() returns a dict for radarsimpy.simulator.sim_radar().
    """
    location : tuple[float, float, float]
    speed    : tuple[float, float, float]
    rcs_dBsm : float
    label    : str
    phase    : float = 0.0

    # ── derived (metadata only, not used by sim_radar) ─────────────────────
    rcs_m2   : float = field(init=False)

    def __post_init__(self):
        self.rcs_m2 = 10 ** (self.rcs_dBsm / 10)

    def to_sim(self) -> dict:
        """Convert to RadarSimPy point-target dict."""
        return {
            "location": self.location,
            "speed"   : self.speed,
            "rcs"     : self.rcs_dBsm,   # API expects dBsm
            "phase"   : self.phase,
        }

    @property
    def R_slant(self) -> float:
        """Slant range from origin (m)."""
        return float(np.sqrt(sum(v**2 for v in self.location)))

    @property
    def az_deg(self) -> float:
        """Azimuth angle (deg), measured from X axis in XY plane."""
        return float(np.degrees(np.arctan2(self.location[1], self.location[0])))

    def print_info(self):
        x, y, z = self.location
        print(f"    Target  : {self.label}")
        print(f"    Position: x={x} m, y={y} m, z={z} m")
        print(f"    Slant R : {self.R_slant:.3f} m")
        print(f"    Azimuth : {self.az_deg:.1f} deg")
        print(f"    RCS     : {self.rcs_m2:.4f} m2  ({self.rcs_dBsm:.1f} dBsm)")


# ════════════════════════════════════════════════════════════════════════════
# Factory functions
# ════════════════════════════════════════════════════════════════════════════

def corner_reflector(
    location: tuple,
    a_leg   : float,
    lam     : float,
    speed   : tuple = (0.0, 0.0, 0.0),
    phase   : float = 0.0,
) -> RadarTarget:
    """
    Trihedral corner reflector.

    RCS = 4*pi*a^4 / (3*lam^2)  (optical region, a >> lam)

    Parameters
    ----------
    location : (x, y, z) in meters
    a_leg    : leg length, m
    lam      : wavelength, m
    """
    rcs_m2   = 4 * np.pi * a_leg**4 / (3 * lam**2)
    rcs_dBsm = 10 * np.log10(rcs_m2)
    label    = f"Corner reflector a={a_leg*100:.0f} cm"
    return RadarTarget(location=location, speed=speed,
                       rcs_dBsm=rcs_dBsm, label=label, phase=phase)


def sphere(
    location: tuple,
    radius  : float,
    lam     : float,
    speed   : tuple = (0.0, 0.0, 0.0),
    phase   : float = 0.0,
) -> RadarTarget:
    """
    Metallic sphere in the optical region (r >> lam).

    RCS = pi*r^2  (lam << r)

    Parameters
    ----------
    location : (x, y, z) in meters
    radius   : sphere radius, m
    lam      : wavelength, m (used to check r >> lam condition)
    """
    if radius < 3 * lam:
        import warnings
        warnings.warn(
            f"sphere: r={radius*1000:.1f} mm < 3*lam={3*lam*1000:.1f} mm — "
            "optical-region formula may be inaccurate (resonance region)",
            UserWarning, stacklevel=2,
        )
    rcs_m2   = np.pi * radius**2
    rcs_dBsm = 10 * np.log10(rcs_m2)
    label    = f"Sphere r={radius*100:.1f} cm"
    return RadarTarget(location=location, speed=speed,
                       rcs_dBsm=rcs_dBsm, label=label, phase=phase)


def point_target(
    location : tuple,
    rcs_dBsm : float,
    label    : str = "Point target",
    speed    : tuple = (0.0, 0.0, 0.0),
    phase    : float = 0.0,
) -> RadarTarget:
    """
    Arbitrary point target with a given RCS.

    Parameters
    ----------
    location : (x, y, z) in meters
    rcs_dBsm : RCS in dBsm (dB relative to 1 m^2)
    """
    return RadarTarget(location=location, speed=speed,
                       rcs_dBsm=rcs_dBsm, label=label, phase=phase)
