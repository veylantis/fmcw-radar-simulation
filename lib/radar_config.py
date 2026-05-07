"""
Global radar configuration for xWR6843AOP
════════════════════════════════════════
Source: TI mmWave Sensing Estimator + SWRA656 + SWRS237
Profile: Short Range Default

RULE: this file is the reference config for all experiments.
If an experiment directory has its own radar_config.py — use that one instead.
"""

import numpy as np

# ── Chirp ────────────────────────────────────────────────────────────────────
f_start_hz = 60.0e9          # Hz  — chirp start frequency
f_slope    = 22.788e12       # Hz/s — FM sweep rate (22.788 MHz/μs)
t_chirp    = 138.73e-6       # s   — ramp duration (ramp end time)
t_idle     = 7e-6            # s   — idle time between chirps
t_cycle    = t_chirp + t_idle          # s   — single TX cycle ≈ 145.73 μs
f_stop_hz  = f_start_hz + f_slope * t_chirp   # Hz  ≈ 63.163 GHz

# ── ADC / Sampling ───────────────────────────────────────────────────────────
fs       = 3e6               # Hz  — sampling rate (3 Msps)
N_cfg    = 394               # samples per config (actual fs*t_chirp=416)

# ── TX/RX ────────────────────────────────────────────────────────────────────
tx_power_dBm  = 12           # dBm — TX power
noise_figure  = 15           # dB  — RX noise figure
rf_gain       = 20           # dB  — RF gain
baseband_gain = 30           # dB  — baseband (IF) gain
load_resistor = 500          # Ohm

# ── TDM ──────────────────────────────────────────────────────────────────────
n_tx         = 3             # number of TX antennas
n_rx         = 4             # number of RX antennas
prp_tdm      = n_tx * t_cycle   # s — PRP for TDM ≈ 437 μs

# ── Derived Parameters ───────────────────────────────────────────────────────
fc        = (f_start_hz + f_stop_hz) / 2    # Hz  — center frequency
lam       = 3e8 / fc                         # m   — wavelength ≈ 4.87 mm
d         = lam / 2                          # m   — array spacing λ/2
bw        = f_stop_hz - f_start_hz           # Hz  — bandwidth ≈ 3.161 GHz
range_res = 3e8 / (2 * bw)                  # m   — range resolution

# ── Antenna Geometry — xWR6843AOP ES2.0 ──────────────────────────────────────
# Positions in meters. Unit: d = λ/2
# TX: TX1(0,0), TX2(4,0), TX3(0,2) in units of λ/2 along (az, el)
tx_channels = [
    {"location": (0, 0 * d, 0 * d), "delay": 0 * t_cycle},   # TX1
    {"location": (0, 4 * d, 0 * d), "delay": 1 * t_cycle},   # TX2
    {"location": (0, 0 * d, 2 * d), "delay": 2 * t_cycle},   # TX3
]

# RX: RX1-4 along azimuth with λ/2 spacing
rx_channels = [
    {"location": (0, 0 * d, 0)},   # RX1
    {"location": (0, 1 * d, 0)},   # RX2
    {"location": (0, 2 * d, 0)},   # RX3
    {"location": (0, 3 * d, 0)},   # RX4
]

# ── Summary Parameters ──────────────────────────────────────────────────────
n_virtual = n_tx * n_rx      # 12 virtual channels

def print_summary():
    """Print a brief summary of the radar configuration."""
    print(f"    Radar       : xWR6843AOP (Short Range Default)")
    print(f"    Bandwidth   : {bw/1e9:.3f} GHz  →  Resolution: {range_res*100:.2f} cm")
    print(f"    fc / λ / d  : {fc/1e9:.2f} GHz / {lam*1000:.2f} mm / {d*1000:.2f} mm")
    print(f"    TX: {n_tx} (TDM)  |  RX: {n_rx}  |  Virtual: {n_virtual}")
    print(f"    PRP (TDM)   : {prp_tdm*1e6:.2f} μs  |  fs: {fs/1e6:.0f} MHz")
