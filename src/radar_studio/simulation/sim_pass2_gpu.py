#!/usr/bin/env python3
"""
Pass 2: GPU Radar Simulation
Generates thousands of pulses to capture target vibration.
"""
import sys, time, json, struct, os
from pathlib import Path
import numpy as np
from scipy import signal
from tqdm import tqdm

# ── Setup Paths ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STUDIO_DIR.parent.parent

# Append PROJECT_ROOT to sys.path so we can import lib
sys.path.append(str(PROJECT_ROOT))

import lib.radar_config as cfg
from radarsimpy import Radar, Receiver, Transmitter
from radarsimpy.simulator import sim_radar

def main():
    print("========================================")
    print(" PASS 2 (GPU): Fast Vibration Pulses")
    print("========================================")
    t_start = time.time()

    # Load Config
    config_path = STUDIO_DIR / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    density = config["simulation"]["gpu_density"]
    n_pulses = config["simulation"]["gpu_pulses"]
    
    loc_vib = np.array(config["scene"]["vibrating_cross"]["location"])
    rot_vib = tuple(config["scene"]["vibrating_cross"]["rotation"])
    f_vib   = config["scene"]["vibrating_cross"]["vibration_freq_hz"]
    a_vib   = config["scene"]["vibrating_cross"]["vibration_amp_m"]
    
    loc_sta = np.array(config["scene"]["static_cross"]["location"])
    rot_sta = tuple(config["scene"]["static_cross"]["rotation"])

    workspace_dir = STUDIO_DIR / "workspace"
    arrays_dir = workspace_dir / "arrays"
    models_dir = workspace_dir / "models"
    scaled_stl = models_dir / "cross_scaled.stl"

    if not scaled_stl.exists():
        print("[!] Error: Scaled STL not found. Run pass 1 first.")
        sys.exit(1)

    print("[1] Setting up Radar Configuration...")
    tx = Transmitter(
        f=[cfg.f_start_hz, cfg.f_stop_hz], t=cfg.t_chirp,
        tx_power=cfg.tx_power_dBm, prp=cfg.prp_tdm, pulses=1,
        channels=cfg.tx_channels
    )
    rx = Receiver(
        fs=cfg.fs, noise_figure=cfg.noise_figure, rf_gain=cfg.rf_gain,
        baseband_gain=cfg.baseband_gain, load_resistor=cfg.load_resistor,
        channels=cfg.rx_channels
    )
    radar = Radar(tx, rx)

    N_adc = int(cfg.fs * cfg.t_chirp)
    n_az = 8

    # Kinematics
    t_axis = np.arange(n_pulses) * cfg.prp_tdm
    y_vib  = loc_vib[1] + a_vib * np.sin(2 * np.pi * f_vib * t_axis)
    vy_vib = a_vib * 2 * np.pi * f_vib * np.cos(2 * np.pi * f_vib * t_axis)

    rwin = signal.windows.chebwin(N_adc, at=60)
    range_profiles = np.zeros((n_pulses, n_az, N_adc), dtype=complex)

    print(f"[2] Running {n_pulses} pulses on GPU (density={density})...")
    
    t_sim = time.time()
    for k in tqdm(range(n_pulses), desc="  GPU pulses", ncols=75):
        tgt_right = {
            "model": str(scaled_stl),
            "location": (float(loc_vib[0]), float(y_vib[k]), float(loc_vib[2])),
            "speed": (0.0, float(vy_vib[k]), 0.0),
            "rotation": rot_vib,
        }
        tgt_left = {
            "model": str(scaled_stl),
            "location": tuple(loc_sta.tolist()),
            "speed": (0.0, 0.0, 0.0),
            "rotation": rot_sta,
        }

        data_k = sim_radar(radar, [tgt_right, tgt_left], density=density, device="gpu")
        bb_k = data_k["baseband"] + data_k["noise"]
        
        # Range FFT
        sig = bb_k[0:n_az, 0, :N_adc]
        range_profiles[k] = np.fft.fft(sig * rwin, n=N_adc, axis=1)

    dt_sim = time.time() - t_sim
    print(f"  OK {n_pulses} pulses in {dt_sim:.1f}s")

    np.save(arrays_dir / "range_profiles_gpu.npy", range_profiles)
    print(f"  Saved range_profiles_gpu.npy")
    
    t_end = time.time()
    print(f"[OK] Pass 2 completed in {t_end - t_start:.2f}s")

if __name__ == "__main__":
    main()
