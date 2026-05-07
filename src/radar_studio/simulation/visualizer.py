#!/usr/bin/env python3
"""
Radar Studio Visualizer
Reads the processed artifacts from `web/data` and generates
high-fidelity diagnostic PNGs in `workspace/plots`.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Setup Paths ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = STUDIO_DIR / "workspace"
WEB_DATA_DIR = STUDIO_DIR / "web" / "data"
PLOT_DIR = WORKSPACE_DIR / "plots"

def log_step(step_name):
    print(f"[{time.strftime('%H:%M:%S')}] ---> {step_name}")

def create_polar_plot(data, r_axis, az_axis, title, filename, is_bartlett=False, R_target=None, theta_tgt=None):
    """
    Renders high-res polar maps like in EXP-014.
    """
    colors = [(0, 0, 0), (0, 0, 0.5), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("RadarPlot", colors, N=256)

    N_adc = len(r_axis)
    max_r = r_axis[-1]
    range_max_plot = 10.0

    if is_bartlett:
        # Interpolate Bartlett to uniform angular grid
        N_polar_az = 720
        uniform_angles_deg = np.linspace(-80, 80, N_polar_az)
        uniform_angles_rad = np.radians(uniform_angles_deg)

        sort_idx = np.argsort(az_axis)
        sa_sorted = np.array(az_axis)[sort_idx]
        data_sorted = data[sort_idx, :]
        uniq = np.concatenate([[True], np.diff(sa_sorted) > 1e-10])
        sa_sorted, data_sorted = sa_sorted[uniq], data_sorted[uniq, :]

        peak_dirty = np.max(data)
        f_interp = interp1d(sa_sorted, data_sorted, axis=0, kind='linear',
                             bounds_error=False, fill_value=peak_dirty - 60)
        data_polar = f_interp(uniform_angles_deg)
        plot_angles = uniform_angles_rad
    else:
        # GT mask is already evenly distributed over 1024 points
        plot_angles = np.radians(np.linspace(-80, 80, data.shape[0]))
        data_polar = data

    arc_theta = np.linspace(np.radians(-80), np.radians(80), 500)
    ula_bounds_rad = np.arcsin(np.linspace(-1, 1, 9))

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)  # CCW
    ax.set_facecolor('black')

    im = ax.pcolormesh(plot_angles, r_axis, data_polar.T, cmap=cm, shading='nearest', zorder=0)
    ax.set_thetamin(-80)
    ax.set_thetamax(80)
    ax.set_ylim(0, range_max_plot)

    # Add vibration target marker if requested
    if R_target is not None and theta_tgt is not None:
        ax.plot(np.radians(theta_tgt), R_target, 'kx', markersize=12, markeredgewidth=1.5, zorder=5, label="Vibration Target")
        ax.legend(loc='upper right')

    # Concentric circles
    for r_m in range(1, int(range_max_plot) + 1):
        lw = 1.2 if r_m % 2 == 0 else 0.5
        ax.plot(arc_theta, np.full_like(arc_theta, r_m), color='black', linewidth=lw, alpha=0.8, zorder=2)

    # Radial lines (only for Bartlett)
    if is_bartlett:
        for a_rad in ula_bounds_rad[1:-1]:
            ax.plot([a_rad, a_rad], [0, range_max_plot], color='black', linewidth=0.5, alpha=0.6, zorder=2)

    ax.set_rgrids([2, 4, 6, 8, 10], labels=['2m', '4m', '6m', '8m', '10m'], fontsize=9, color='white', alpha=0.8)
    ax.set_thetagrids([])
    for la in [-60, -30, 0, 30, 60]:
        ax.text(np.radians(la), range_max_plot * 1.12, f"{la}°", ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=30)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Power (dB)' if is_bartlett else 'Hit Value')
    
    fig.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    print("========================================")
    print(" DIAGNOSTIC VISUALIZER")
    print("========================================")
    t_start = time.time()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    axes_path = WEB_DATA_DIR / "axes.json"
    if not axes_path.exists():
        print("[!] axes.json not found, cannot plot.")
        sys.exit(1)

    with open(axes_path, "r") as f:
        axes = json.load(f)

    r_axis = np.array(axes["range_axis"])
    az_axis = np.array(axes["azimuth_axis"])

    # Extract target location from vibration data if it exists
    R_target = None
    theta_tgt = None
    vib_path = WEB_DATA_DIR / "vibration_data.json"
    if vib_path.exists():
        with open(vib_path, "r") as f:
            vib_data = json.load(f)
            R_target = vib_data["cell"]["r_m"]
            theta_tgt = vib_data["cell"]["az_deg"]

    # 1. Bartlett Map
    bartlett_path = WEB_DATA_DIR / "bartlett_map.npy"
    if bartlett_path.exists():
        log_step("Plotting Bartlett Polar Map")
        bartlett = np.load(bartlett_path)
        create_polar_plot(
            bartlett, r_axis, az_axis, 
            "Bartlett — Polar (CPU Raytracing)", "bartlett_polar.png", 
            is_bartlett=True, R_target=R_target, theta_tgt=theta_tgt
        )

    # 2. Ground Truth Map
    gt_path = WEB_DATA_DIR / "gt_map.npy"
    if gt_path.exists():
        log_step("Plotting GT Polar Map")
        gt = np.load(gt_path)
        create_polar_plot(
            gt, r_axis, az_axis, 
            "Ground Truth — Polar Mask", "gt_polar.png", 
            is_bartlett=False
        )

    # 3. Vibration Data
    if vib_path.exists():
        log_step("Plotting Vibration Analysis")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(vib_data["time_axis"], vib_data["displacement_um"], 'b')
        ax1.set_title("Target Displacement")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Displacement (µm)")
        ax1.grid(True)
        
        ax2.plot(vib_data["freq_axis"], vib_data["spectrum_db"], 'r')
        ax2.set_title("Vibration Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlim(0, 100)
        ax2.grid(True)
        
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "vibration_analysis.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("========================================")
    print(f"[SUCCESS] Visualizations saved in {time.time() - t_start:.2f}s")
    print("========================================")

if __name__ == "__main__":
    main()
