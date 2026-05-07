#!/usr/bin/env python3
"""
Pass 1: CPU Raytracing and Baseband Generation
Generates a 1-pulse snapshot of the scene, logging the ray paths to HDF5.
"""
import sys, time, json, struct, os
from pathlib import Path
import numpy as np

# ── Setup Paths ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STUDIO_DIR.parent.parent

# Append PROJECT_ROOT to sys.path so we can import lib,
# without overriding the radarsimpy module set by PYTHONPATH.
sys.path.append(str(PROJECT_ROOT))

import lib.radar_config as cfg
from radarsimpy import Radar, Receiver, Transmitter
from radarsimpy.simulator import sim_radar

def load_binary_stl(path, scale=1.0, offset=(0,0,0)):
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        normals  = np.zeros((n_tri, 3))
        vertices = np.zeros((n_tri, 3, 3))
        for i in range(n_tri):
            data = struct.unpack('<12fH', f.read(50))
            normals[i] = data[0:3]
            for j in range(3):
                vertices[i, j] = [
                    data[3 + j*3 + 0] * scale + offset[0],
                    data[3 + j*3 + 1] * scale + offset[1],
                    data[3 + j*3 + 2] * scale + offset[2],
                ]
        for i in range(n_tri):
            e1 = vertices[i,1] - vertices[i,0]
            e2 = vertices[i,2] - vertices[i,0]
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if normals[i].sum() == 0 and nl > 0:
                normals[i] = n / nl
    return vertices, normals

def main():
    print("========================================")
    print(" PASS 1 (CPU): Raytracing & 1-Pulse")
    print("========================================")
    t_start = time.time()

    # Load Config
    config_path = STUDIO_DIR / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    density = config["simulation"]["cpu_density"]
    loc_vib = config["scene"]["vibrating_cross"]["location"]
    rot_vib = config["scene"]["vibrating_cross"]["rotation"]
    loc_sta = config["scene"]["static_cross"]["location"]
    rot_sta = config["scene"]["static_cross"]["rotation"]

    workspace_dir = STUDIO_DIR / "workspace"
    ray_log_dir = workspace_dir / "raytrace_log"
    arrays_dir = workspace_dir / "arrays"
    models_dir = workspace_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure ray_log_dir is clean
    if ray_log_dir.exists():
        import shutil
        shutil.rmtree(ray_log_dir)
    ray_log_dir.mkdir(parents=True, exist_ok=True)

    # Prepare STL
    print("[1] Preparing scaled STL mesh...")
    STL_PATH = PROJECT_ROOT / "models" / "curated" / "cross.stl"
    scaled_stl = models_dir / "cross_scaled.stl"

    if not scaled_stl.exists():
        verts, norms = load_binary_stl(STL_PATH, scale=2.5, offset=(0,0,0))
        with open(scaled_stl, 'wb') as f:
            f.write(b'\0' * 80)
            f.write(struct.pack('<I', len(verts)))
            for i in range(len(verts)):
                f.write(struct.pack('<3f', *norms[i]))
                for j in range(3):
                    f.write(struct.pack('<3f', *verts[i,j]))
                f.write(struct.pack('<H', 0))
        print(f"    Saved scaled STL to workspace.")

    # Radar Setup
    print("[2] Setting up Radar Configuration...")
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

    # Targets
    tgt_right = {
        "model": str(scaled_stl),
        "location": tuple(loc_vib),
        "speed": (0.0, 0.0, 0.0),
        "rotation": tuple(rot_vib),
    }
    tgt_left = {
        "model": str(scaled_stl),
        "location": tuple(loc_sta),
        "speed": (0.0, 0.0, 0.0),
        "rotation": tuple(rot_sta),
    }

    print(f"[3] Running sim_radar on CPU (density={density})...")
    
    data = sim_radar(
        radar, 
        [tgt_right, tgt_left], 
        density=density, 
        device="cpu",
        log_path=str(ray_log_dir)
    )

    baseband = data["baseband"] + data["noise"]
    
    np.save(arrays_dir / "baseband_cpu.npy", baseband)
    
    t_end = time.time()
    print(f"[OK] Pass 1 completed in {t_end - t_start:.2f}s")
    print(f"     Baseband saved to arrays/baseband_cpu.npy")
    print(f"     Ray logs saved to workspace/raytrace_log/")

if __name__ == "__main__":
    main()
