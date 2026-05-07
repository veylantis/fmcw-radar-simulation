#!/usr/bin/env python3
"""
DSP Processor for Radar Studio
Processes Pass 1 (CPU) baseband and Ray logs.
Processes Pass 2 (GPU) range profiles for vibration.
Exports JSON/NPY for the web dashboard.
"""
import sys, time, json, os
from pathlib import Path
import numpy as np
from scipy import signal
from tqdm import tqdm

# ── Setup Paths ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STUDIO_DIR.parent.parent

sys.path.append(str(PROJECT_ROOT))

import lib.radar_config as cfg
import lib.ideal_gt as ideal_gt

def log_step(step_name):
    print(f"\n[{time.strftime('%H:%M:%S')}] ---> {step_name}")

def main():
    print("========================================")
    print(" DSP PROCESSOR: Aggregating Data")
    print("========================================")
    t_start = time.time()

    # Load Config
    log_step("Loading configuration")
    config_path = STUDIO_DIR / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    workspace_dir = STUDIO_DIR / "workspace"
    arrays_dir = workspace_dir / "arrays"
    ray_log_dir = workspace_dir / "raytrace_log"
    web_data_dir = STUDIO_DIR / "web" / "data"
    web_data_dir.mkdir(parents=True, exist_ok=True)

    N_adc = int(cfg.fs * cfg.t_chirp)
    max_r = 3e8 * N_adc / (2 * cfg.bw)
    N_fft = 1024
    
    sin_theta = np.fft.fftshift(np.fft.fftfreq(N_fft, d=0.5))
    scan_angles = np.degrees(np.arcsin(np.clip(-sin_theta, -1, 1)))

    loc_vib = config["scene"]["vibrating_cross"]["location"]
    R_target = np.linalg.norm(loc_vib)
    theta_tgt = np.degrees(np.arctan2(loc_vib[1], loc_vib[0]))
    
    # ── 1. Process CPU Data (Bartlett & GT) ───────────────
    bb_cpu_path = arrays_dir / "baseband_cpu.npy"
    if bb_cpu_path.exists():
        log_step("[1] Processing CPU Baseband for Bartlett Map")
        t0 = time.time()
        bb_cpu = np.load(bb_cpu_path)
        sig = bb_cpu[0:8, 0, :N_adc]
        rwin = signal.windows.chebwin(N_adc, at=60)
        range_profile = np.fft.fft(sig * rwin, n=N_adc, axis=1)

        spatial_win = signal.windows.chebwin(8, at=50)
        ra_map_complex = np.zeros((N_fft, N_adc), dtype=complex)
        
        print("    Computing Spatial FFT (Bartlett)...")
        for r in tqdm(range(N_adc), desc="    Range Bins", ncols=75):
            y = range_profile[0:8, r]
            ra_map_complex[:, r] = np.fft.fftshift(np.fft.fft(y * spatial_win, n=N_fft))
            
        ra_map_dirty_db = 20 * np.log10(np.abs(ra_map_complex) + 1e-12)
        peak_dirty = np.max(ra_map_dirty_db)
        ra_map_dirty_db = np.clip(ra_map_dirty_db, peak_dirty - 60, peak_dirty)
        np.save(web_data_dir / "bartlett_map.npy", ra_map_dirty_db)
        print(f"    [OK] Bartlett map saved in {time.time()-t0:.2f}s")
    else:
        print("[!] CPU Baseband not found. Skipping Bartlett Map.")

    log_step("[2] Parsing Raytrace Log for GT Map and 3D Rays")
    if ray_log_dir.exists():
        t0 = time.time()
        hit_r, hit_az, hit_power, hit_bounces, hit_paths, hit_refs = ideal_gt.parse_raypool(ray_log_dir, return_paths=True)
        if len(hit_r) > 0:
            print("    Generating Ideal GT Mask...")
            gt_mask = ideal_gt.generate_ideal_mask(
                hit_r, hit_az, hit_power, 
                n_angles=1024, angle_min=-80.0, angle_max=80.0,
                n_range=N_adc, range_min=0.0, range_max=max_r
            )
            np.save(web_data_dir / "gt_map.npy", gt_mask)
            
            print("    Exporting 3D Ray Paths for visualization...")
            max_rays_export = 500
            subset_indices = np.argsort(hit_power)[::-1][:max_rays_export]
            export_paths = []
            for i in tqdm(subset_indices, desc="    Extracting Rays", ncols=75):
                path = hit_paths[i]
                ref_count = hit_refs[i]
                valid_pts = path[:int(ref_count)+1]
                export_paths.append(valid_pts.tolist())
                
            with open(web_data_dir / "ray_paths.json", "w") as f:
                json.dump({"rays": export_paths}, f)
            print(f"    [OK] GT Map and {len(export_paths)} ray paths exported in {time.time()-t0:.2f}s.")
        else:
            print("    [!] No rays found in log.")
    else:
        print("[!] Raytrace log not found. Skipping GT Map.")

    # ── 2. Process GPU Data (Vibration) ───────────────────
    rp_gpu_path = arrays_dir / "range_profiles_gpu.npy"
    if rp_gpu_path.exists():
        log_step("[3] Processing GPU Range Profiles for Vibration Analysis")
        t0 = time.time()
        range_profiles = np.load(rp_gpu_path)
        
        # Target Extraction Logic
        range_bin = int(round(R_target / (max_r / N_adc)))
        angle_bin = int(np.argmin(np.abs(scan_angles - theta_tgt)))
        
        print(f"    Target Extracted at Range Bin {range_bin}, Angle Bin {angle_bin} ({R_target:.2f}m, {theta_tgt:.1f}°)")
        
        # Beamform at target cell
        print("    Performing Beamforming across time history...")
        y_all = range_profiles[:, :, range_bin] # [N_PULSES, 8]
        spatial_win = signal.windows.chebwin(8, at=50)
        bf_all = np.fft.fftshift(
            np.fft.fft(y_all * spatial_win[np.newaxis, :], n=N_fft, axis=1),
            axes=1
        )
        complex_history = bf_all[:, angle_bin]
        
        print("    Unwrapping phase and recovering displacement...")
        phase_raw = np.angle(complex_history)
        phase_unwrapped = np.unwrap(phase_raw)
        
        n_pulses = len(phase_raw)
        t_axis = np.arange(n_pulses) * cfg.prp_tdm
        
        # Detrending
        coeffs = np.polyfit(t_axis, phase_unwrapped, deg=1)
        phase_detrended = phase_unwrapped - np.polyval(coeffs, t_axis)
        
        displacement_recovered = phase_detrended * cfg.lam / (4 * np.pi)
        
        # Spectrum
        print("    Computing FFT Spectrum of vibration...")
        spec_win = signal.windows.hann(n_pulses)
        disp_windowed = displacement_recovered * spec_win
        freq_axis = np.fft.rfftfreq(n_pulses, d=cfg.prp_tdm)
        spectrum = np.abs(np.fft.rfft(disp_windowed))
        spectrum_db = 20 * np.log10(spectrum / (spectrum.max() + 1e-30) + 1e-30)
        
        vib_data = {
            "time_axis": t_axis.tolist(),
            "displacement_um": (displacement_recovered * 1e6).tolist(),
            "freq_axis": freq_axis.tolist(),
            "spectrum_db": spectrum_db.tolist(),
            "cell": {"range_bin": range_bin, "angle_bin": angle_bin, "r_m": R_target, "az_deg": theta_tgt}
        }
        with open(web_data_dir / "vibration_data.json", "w") as f:
            json.dump(vib_data, f)
        print(f"    [OK] Vibration data saved in {time.time()-t0:.2f}s")
    else:
        print("[!] GPU Range profiles not found. Skipping Vibration Analysis.")

    log_step("Saving Reference Axes")
    axes_data = {
        "range_axis": np.linspace(0, max_r, N_adc).tolist(),
        "azimuth_axis": scan_angles.tolist(),
        "max_range": float(max_r)
    }
    with open(web_data_dir / "axes.json", "w") as f:
        json.dump(axes_data, f)
        
    # ── 3. Diagnostic Plots ───────────────────────────────

    t_end = time.time()
    print("========================================")
    print(f"[SUCCESS] DSP Processing fully completed in {t_end - t_start:.2f}s")
    print("========================================")

if __name__ == "__main__":
    main()
