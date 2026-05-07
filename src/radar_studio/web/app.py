import sys, json, time
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
STUDIO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STUDIO_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
import lib.ideal_gt as ideal_gt

app = FastAPI()

# Mount frontend files
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR / "static")), name="static")

# Mount workspace directory to serve the scaled STL
WORKSPACE_DIR = STUDIO_DIR / "workspace"
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")

# Global DB loaded on startup
DB = {
    "bartlett_map": None,
    "gt_map": None,
    "vibration_data": None,
    "rays": {},
    "scene": None,
    "r_axis": None,
    "az_axis": None,
}

@app.on_event("startup")
def load_data():
    t0 = time.time()
    
    # Load configuration
    config_path = STUDIO_DIR / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            DB["scene"] = json.load(f)["scene"]
    else:
        print("[!] Warning: config.json not found!")

    # Load data artifacts
    web_data_dir = SCRIPT_DIR / "data"
    
    axes_path = web_data_dir / "axes.json"
    if axes_path.exists():
        with open(axes_path, "r") as f:
            axes = json.load(f)
            DB["r_axis"] = axes["range_axis"]
            DB["az_axis"] = axes["azimuth_axis"]
            
    bartlett_path = web_data_dir / "bartlett_map.npy"
    if bartlett_path.exists() and axes_path.exists():
        bartlett_raw = np.load(bartlett_path)
        # Interpolate Bartlett to uniform 1024 grid to match GT and Hover-Sync
        uniform_angles = np.linspace(-80, 80, 1024)
        az_raw = np.array(axes["azimuth_axis"])
        sort_idx = np.argsort(az_raw)
        sa_sorted = az_raw[sort_idx]
        bart_sorted = bartlett_raw[sort_idx, :]
        
        uniq = np.concatenate([[True], np.diff(sa_sorted) > 1e-10])
        sa_sorted = sa_sorted[uniq]
        bart_sorted = bart_sorted[uniq, :]
        
        f_interp = interp1d(sa_sorted, bart_sorted, axis=0, kind='linear', bounds_error=False, fill_value=np.min(bart_sorted))
        bartlett_uniform = f_interp(uniform_angles)
        
        DB["bartlett_map"] = bartlett_uniform.tolist()
        DB["az_axis"] = uniform_angles.tolist() # Replace with uniform axis!
        
    gt_path = web_data_dir / "gt_map.npy"
    if gt_path.exists():
        DB["gt_map"] = np.load(gt_path).tolist()
        
    vib_path = web_data_dir / "vibration_data.json"
    if vib_path.exists():
        with open(vib_path, "r") as f:
            DB["vibration_data"] = json.load(f)

    # Rebuild Ray DB for Hover-Sync
    print("Prepping GT Database... Loading Raytrace Log...")
    log_dir = WORKSPACE_DIR / "raytrace_log"
    if log_dir.exists():
        hit_r, hit_az, hit_pw, hit_bnc, hit_paths, hit_refs = ideal_gt.parse_raypool(log_dir, return_paths=True)
        print(f"Raypool parsed in {time.time()-t0:.2f}s")
        
        t1 = time.time()
        
        if DB["r_axis"] is not None and DB["az_axis"] is not None:
            N_adc = len(DB["r_axis"])
            # Use exact max_range from dsp_processor to match generate_ideal_mask bins
            if "max_range" in axes:
                max_r = axes["max_range"]
            else:
                dr = DB["r_axis"][1] - DB["r_axis"][0] if N_adc > 1 else DB["r_axis"][0] * 2
                max_r = DB["r_axis"][-1] + dr / 2.0
            r_bins = np.linspace(0, max_r, N_adc + 1)
            
            az_bins = np.linspace(-80, 80, 1024+1) # 1024 points for GT
            
            idx_az = np.searchsorted(az_bins, hit_az) - 1
            idx_r = np.searchsorted(r_bins, hit_r) - 1
            
            valid_mask = (idx_az >= 0) & (idx_az < 1024) & (idx_r >= 0) & (idx_r < N_adc)
            idx_az = idx_az[valid_mask]
            idx_r = idx_r[valid_mask]
            hit_pw = hit_pw[valid_mask]
            hit_paths = hit_paths[valid_mask]
            hit_refs = hit_refs[valid_mask]
            
            print(f"Indexing {len(hit_pw)} valid bounds-checked hits for Hover-Sync...")
            linear = idx_az * N_adc + idx_r
            sort_i = np.lexsort((-hit_pw, linear))
            
            sorted_lin = linear[sort_i]
            if len(sorted_lin) > 0:
                changes = np.where(sorted_lin[:-1] != sorted_lin[1:])[0] + 1
                splits = np.split(sort_i, changes)
                for g_indices in splits:
                    if len(g_indices) == 0: continue
                    
                    best_indices = g_indices[:20]  # Limit to 20 paths maximum per pixel
                    lin_val = linear[best_indices[0]]
                    a = lin_val // N_adc
                    r = lin_val % N_adc
                    
                    paths_export = []
                    for i in best_indices:
                        rc = hit_refs[i]
                        pts = [[0.0, 0.0, 0.0]]
                        # Location[0] = radar origin [0,0,0], bounces at Location[1..rc]
                        for pt_idx in range(1, int(rc) + 1):
                            pts.append(hit_paths[i, pt_idx].tolist())
                        pts.append([0.0, 0.0, 0.0])
                        paths_export.append(pts)
                    
                    DB["rays"][f"{a}_{r}"] = paths_export

            print(f"Ray index complete by {time.time()-t1:.2f}s!")
    else:
        print("[!] Raytrace log not found. Hover-sync will be empty.")

@app.get("/")
def home():
    return FileResponse(SCRIPT_DIR / "static" / "index.html")

@app.get("/api/scene")
def get_scene():
    if DB["scene"]:
        return JSONResponse({"targets": list(DB["scene"].values())})
    return JSONResponse({})

@app.get("/api/maps")
def get_maps():
    return JSONResponse({
        "bartlett": DB["bartlett_map"],
        "gt": DB["gt_map"],
        "r": DB["r_axis"],
        "az": DB["az_axis"]
    })

@app.get("/api/vibration")
def get_vibration():
    return JSONResponse(DB["vibration_data"] if DB["vibration_data"] else {})

@app.get("/api/rays")
def get_rays(a: int, r: int):
    k = f"{a}_{r}"
    if k in DB["rays"]:
        return JSONResponse({"paths": DB["rays"][k]})
    return JSONResponse({"paths": []})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
