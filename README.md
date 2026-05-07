# FMCW Radar Studio — Ray Tracing Simulation & Vibration Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![RadarSimPy](https://img.shields.io/badge/RadarSimPy-Required-orange)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688?logo=fastapi&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-3D%20Visualization-black?logo=three.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<p align="center">
<img width="1200" height="614" alt="Recording 2026-05-06 173007" src="https://github.com/user-attachments/assets/4b7c47b2-7ed2-4b8b-9f1d-78b539a0310a" />
</p>

Validate radar configs, prototype DSP, and generate training data without hardware. End-to-end 60 GHz FMCW pipeline with physical ray tracing, MIMO beamforming, and vital-signs-grade vibration recovery — all behind an interactive web dashboard.
End-to-end 60 GHz FMCW radar simulation built on top of [RadarSimPy](https://radarsimx.com). Models a TI xWR6843AOP radar with 3TX/4RX MIMO, performs CPU-based ray tracing with full bounce logging, GPU-accelerated pulse capture for vibration sensing, and serves the results through an interactive web dashboard with real-time 3D ray inspection.

<!-- ⬇️ PLACEHOLDER: Upload a wide hero screenshot or GIF of the web dashboard here -->
<!-- ![Radar Studio Dashboard](docs/images/dashboard_hero.png) -->

---

## What This Does

The vibration recovery pipeline demonstrates the same DSP foundation used for non-contact vital signs monitoring (heart rate, respiration) on mmWave radar.
The pipeline simulates a two-target scene (one vibrating, one static) using physical ray tracing, then processes the raw baseband through a complete DSP chain:

1. **Pass 1 — CPU Ray Tracing**: Fires rays at STL mesh targets, logs every bounce path to HDF5, generates a 1-pulse baseband snapshot.
2. **Pass 2 — GPU Pulse Capture**: Runs 2500 chirps on GPU to build a slow-time history of the vibrating target's micro-displacement.
3. **DSP Processing**: Produces Bartlett beamforming maps, ray-traced ground truth masks, and vibration spectral analysis.
4. **Web Dashboard**: FastAPI + Three.js + Plotly interface where hovering over any pixel in the ground truth map renders the corresponding 3D ray paths in real time.

## Who this is for

1. RF engineers validating chirp configs before silicon arrives.
2. ML teams generating synthetic radar data.
3. Biomedical researchers prototyping non-contact vital signs.
4. Students learning mmWave DSP end-to-end.

### Pipeline Output Examples

<table>
  <tr>
    <td align="center"><strong>Bartlett Beamforming (CPU Raytracing)</strong></td>
    <td align="center"><strong>Ground Truth Mask (Ray-Traced)</strong></td>
  </tr>
  <tr>
    <td>
      <!-- ⬇️ PLACEHOLDER: Upload bartlett_polar.png here -->
      <img width="1655" height="1500" alt="bartlett_polar" src="https://github.com/user-attachments/assets/2dd9a436-f280-4a21-bd7a-5639d0efcea0" />
    </td>
    <td>
      <!-- ⬇️ PLACEHOLDER: Upload gt_polar.png here -->
      <img width="1655" height="1500" alt="gt_polar" src="https://github.com/user-attachments/assets/84ff7aac-6a41-44ff-a6e5-95519e55ce9d" />
    </td>
  </tr>
</table>

<!-- ⬇️ PLACEHOLDER: Upload vibration_analysis.png here -->
<p align="center">
  <img width="2780" height="1577" alt="vibration_recovered" src="https://github.com/user-attachments/assets/caee7770-a043-48e8-b523-cef87bdb5fef" />
</p>
<p align="center"><em>Target displacement recovery (5 Hz)</em></p>
<p align="center">
  <img width="2781" height="1577" alt="vibration_spectrum" src="https://github.com/user-attachments/assets/1f8efc66-3a73-4941-ac5d-929ad15723d3" />
</p>
<p align="center">FFT spectrum confirming the vibration frequency.</em></p>

---

## Radar Configuration

Models the **TI xWR6843AOP** (60–64 GHz, Antenna-on-Package):

| Parameter | Value |
|-----------|-------|
| Frequency | 60.0 – 63.2 GHz |
| Bandwidth | ~3.16 GHz |
| Range Resolution | ~4.7 cm |
| TX / RX | 3 / 4 (12 virtual channels, TDM-MIMO) |
| ADC Sampling | 3 Msps, 416 samples/chirp |
| PRP (TDM) | ~437 µs |

Full derivation with antenna geometry and TI profileCfg parameters: [`radar_config.md`](radar_config.md)

---

## Project Structure

```
fmcw-radar-simulation/
├── lib/                          # Core radar engine
│   ├── radar_config.py           # xWR6843AOP hardware parameters
│   ├── radar_targets.py          # Target placement & vibration model
│   ├── radar_viz.py              # Plotly/Matplotlib visualization utilities
│   └── ideal_gt.py               # HDF5 ray log parser & ground truth generator
├── src/radar_studio/             # Studio orchestrator
│   ├── run_studio.py             # Pipeline entry point (runs all passes)
│   ├── config.json               # Scene + simulation parameters
│   ├── simulation/
│   │   ├── sim_pass1_cpu.py      # CPU ray tracing pass
│   │   ├── sim_pass2_gpu.py      # GPU vibration pulse capture
│   │   ├── dsp_processor.py      # Beamforming, GT mask, vibration DSP
│   │   └── visualizer.py         # Diagnostic plot generation
│   └── web/
│       ├── app.py                # FastAPI dashboard server
│       └── static/index.html     # Three.js + Plotly frontend
├── models/curated/cross.stl      # Target 3D mesh (cross shape)
├── envs/                         # RadarSimPy binaries (not tracked)
│   ├── cpu/radarsimpy/           # CPU build (compiled extensions)
│   └── gpu/radarsimpy/           # GPU/CUDA build (compiled extensions)
├── radar_config.md               # Detailed radar parameter reference
└── requirements.txt
```

---

## Prerequisites

- **Python 3.10–3.12**
- **NVIDIA GPU + CUDA** (for Pass 2; Pass 1 runs on CPU only)
- **RadarSimPy** pre-compiled binaries from [radarsimx.com](https://radarsimx.com)

> RadarSimPy is not pip-installable. It ships as platform-specific compiled extensions (`.pyd` on Windows, `.so` on Linux) and must be placed manually. The simulation requires two separate builds — one for CPU ray tracing and one for GPU (CUDA) pulse generation.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/veylantis/fmcw-radar-simulation.git
cd fmcw-radar-simulation
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
source .venv/Scripts/activate    # Windows (Git Bash)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up RadarSimPy

Download both CPU and GPU builds from [radarsimx.com](https://radarsimx.com). Each download contains a `radarsimpy/` folder with compiled extensions for your platform.

Place them so the directory structure looks like:

```
envs/
├── cpu/
│   └── radarsimpy/
│       ├── __init__.py
│       ├── radarsimpy.*        # CPU-only build
│       └── ...
└── gpu/
    └── radarsimpy/
        ├── __init__.py
        ├── radarsimpy.*        # CUDA-enabled build
        └── ...
```

You can copy the folders directly, or symlink if you already have them elsewhere:

```bash
# Linux / macOS
ln -s /path/to/your/radarsimpy-cpu envs/cpu
ln -s /path/to/your/radarsimpy-gpu envs/gpu
```

```powershell
# Windows (PowerShell, run as admin)
New-Item -ItemType Junction -Path envs\cpu -Target "X:\path\to\radarsimpy-cpu"
New-Item -ItemType Junction -Path envs\gpu -Target "X:\path\to\radarsimpy-gpu"
```

### 5. Configure paths

Edit `src/radar_studio/config.json` and set the `env_path_cpu` and `env_path_gpu` fields to absolute paths on your system:

```json
{
    "simulation": {
        "env_path_cpu": "/absolute/path/to/envs/cpu",
        "env_path_gpu": "/absolute/path/to/envs/gpu"
    }
}
```

The orchestrator prepends these to `PYTHONPATH` at runtime so that each simulation pass imports the correct RadarSimPy build without conflicts.

---

## Usage

### Run the full pipeline

```bash
python src/radar_studio/run_studio.py
```

This executes all four phases sequentially. Output goes to `src/radar_studio/workspace/` (arrays, ray logs, plots) and `src/radar_studio/web/data/` (dashboard artifacts).

Expect approximately:
- Pass 1 (CPU): 2–5 min depending on `cpu_density`
- Pass 2 (GPU): 5–20 min depending on `gpu_pulses` and GPU model
- DSP + Visualizer: < 30 sec

### Launch the web dashboard

After the pipeline completes:

```bash
cd src/radar_studio/web
python app.py
```

Open **http://127.0.0.1:8000** in your browser.

The dashboard loads the ground truth heatmap on the left and the 3D scene on the right. Hover over any pixel in the heatmap to render the corresponding ray bounce paths in real time.

---

## Scene Configuration

Edit `src/radar_studio/config.json` to modify the scene:

```json
{
    "scene": {
        "vibrating_cross": {
            "location": [2.5, -1.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "vibration_freq_hz": 5.0,
            "vibration_amp_m": 0.002
        },
        "static_cross": {
            "location": [2.5, 0.0, 0.0],
            "rotation": [30.0, 45.0, 15.0]
        }
    },
    "simulation": {
        "cpu_density": 0.3,
        "gpu_density": 0.1,
        "gpu_pulses": 2500
    }
}
```

- **`cpu_density`** / **`gpu_density`** — ray density per wavelength². Higher = more accurate but slower.
- **`gpu_pulses`** — number of slow-time samples. Controls vibration frequency resolution (PRF / N_pulses).

---

## How It Works

### Ray Tracing & Ground Truth

Pass 1 fires rays from the radar origin toward the scene meshes using RadarSimPy's CPU solver. Every ray-surface intersection (including multi-bounce reflections) is logged to HDF5 files with:

- Hit coordinates (range, azimuth)
- Received power (path loss + RCS)
- Full 3D bounce path geometry

The DSP processor then bins these hits into a range-azimuth grid to produce the **ground truth mask** — a noise-free reference map showing exactly where the targets are, based purely on geometry.

### Bartlett Beamforming

The same 1-pulse baseband from Pass 1 goes through range FFT (Chebyshev-windowed) followed by spatial FFT across the 8 azimuth virtual channels. This produces the conventional **Bartlett power map** — what the radar actually "sees" through its array processing, including sidelobes and resolution limits.

### Vibration Recovery

Pass 2 captures 2500 consecutive pulses with the vibrating target oscillating at a configured frequency. The DSP pipeline:

1. Beamforms each pulse at the target's range-angle cell
2. Extracts the complex phase history
3. Unwraps and detrends the phase
4. Converts phase to displacement via `Δr = Δφ · λ / (4π)`
5. Computes the FFT spectrum to identify vibration frequency

---

## License

This project is licensed under the [MIT License](LICENSE).  
RadarSimPy is a separate product — see [radarsimx.com](https://radarsimx.com) for its licensing terms.
