#!/usr/bin/env python3
"""
Radar Studio Orchestrator
Executes Pass 1 (CPU), Pass 2 (GPU), and DSP processing.
"""
import sys, os, time, subprocess, json
from pathlib import Path

# ── Setup Paths ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR
SIM_DIR = STUDIO_DIR / "simulation"

def run_script(script_path, env_path=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_path:
        # Prepend the specialized env path to PYTHONPATH
        env["PYTHONPATH"] = str(env_path) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, str(script_path)]
    print(f"\n========================================")
    print(f"[{time.strftime('%H:%M:%S')}] Executing Phase: {script_path.name}")
    print(f"[{time.strftime('%H:%M:%S')}] Python Interpreter: {sys.executable}")
    print(f"[{time.strftime('%H:%M:%S')}] PYTHONPATH prefix: {env_path if env_path else 'None'}")
    print(f"========================================")
    
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n[!] Error: {script_path.name} failed with exit code {result.returncode} after {time.time()-t0:.2f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n[OK] Phase {script_path.name} completed successfully in {time.time()-t0:.2f}s")

def main():
    print("========================================")
    print(" RADAR STUDIO ORCHESTRATOR")
    print("========================================")
    t_start = time.time()

    # Load Config
    config_path = STUDIO_DIR / "config.json"
    if not config_path.exists():
        print("[!] config.json not found.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = json.load(f)

    env_cpu = Path(config["simulation"]["env_path_cpu"]).resolve()
    env_gpu = Path(config["simulation"]["env_path_gpu"]).resolve()

    if not env_cpu.exists() or not env_gpu.exists():
        print("[!] CPU or GPU environment paths not found. Please create the junctions.")
        sys.exit(1)
    
    # 1. Pass 1: CPU Raytracing & Baseband
    run_script(SIM_DIR / "sim_pass1_cpu.py", env_path=env_cpu)

    # 2. Pass 2: GPU Fast Vibration Capture
    run_script(SIM_DIR / "sim_pass2_gpu.py", env_path=env_gpu)

    # 3. DSP Processing (Environment neutral, uses standard libs)
    run_script(SIM_DIR / "dsp_processor.py")

    # 4. Visualization (Generate Diagnostic PNGs)
    run_script(SIM_DIR / "visualizer.py")

    t_end = time.time()
    print("\n========================================")
    print(f" PIPELINE COMPLETED IN {t_end - t_start:.2f}s")
    print("========================================")
    print("Ready for Web Dashboard.")

if __name__ == "__main__":
    main()
