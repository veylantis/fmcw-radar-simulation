import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def parse_raypool(log_dir, radar_pos=(0, 0, 0), return_paths=False):
    """
    Parses a radarsimpy raypool HDF5 directory, extracts valid physical scattering points
    from ALL occupancy sector files, using fully vectorized numpy operations.
    
    If return_paths is True, returns additionally the FULL Location history for each valid ray point 
    so the Web Explorer can visualize the entire bounce trace.
    
    Returns:
        hit_r: numpy array of unwrapped multipath ranges (in meters)
        hit_az: numpy array of azimuth angles FROM radar TO target (in degrees)
        hit_power: numpy array of accumulated ray power scattered back (with 1/R^2 mapping loss)
        hit_bounces: numpy array of the number of bounces each extracted ray experienced
        [Optional] hit_paths: numpy array of shape (N, 11, 3) tracking the ray history from radar.
        [Optional] hit_refs: numpy array of shape (N,) tracking where the ray hit stopped.
    """
    radar_pos = np.array(radar_pos)
    all_hit_r, all_hit_az, all_hit_power, all_hit_bounces = [], [], [], []
    if return_paths:
        all_hit_paths, all_hit_refs = [], []
    
    h5_files = glob.glob(str(Path(log_dir) / "raypool_snapshot_*_occupancy_sector_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No raypool files found in {log_dir}")
        
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            if 'H5RaypoolStruct' not in f:
                continue
            ds = f['H5RaypoolStruct'][:]
            
        ref_counts = ds['RefCount']  
        valid_mask = ref_counts > 0
        if not np.any(valid_mask):
            continue
            
        ds_valid = ds[valid_mask]    
        valid_refs = ds_valid['RefCount'] 
        
        locs = ds_valid['Location']      
        fwd = ds_valid['Range']          
        pol = ds_valid['Polarization']   
        
        N_valid = len(ds_valid)
        bounce_indices = np.arange(11)[np.newaxis, :]
        mask_bounces = (bounce_indices > 0) & (bounce_indices <= valid_refs[:, np.newaxis])
        
        locs_valid = locs[mask_bounces] 
        fwd_valid = fwd[mask_bounces]   
        pol_valid = pol[mask_bounces]   
        bounces_flat = np.broadcast_to(bounce_indices, (N_valid, 11))[mask_bounces] 
        
        v_to_target = locs_valid - radar_pos
        back_R = np.linalg.norm(v_to_target, axis=-1)
        r_event = 0.5 * (fwd_valid + back_R)
        
        # Verify Sanity
        # For bounce=1, r_event must equal back_R within ray engine epsilon (up to ~2.5mm seen)
        eps = 1e-2
        b1_mask = bounces_flat == 1
        if np.any(b1_mask):
            assert np.allclose(r_event[b1_mask], back_R[b1_mask], atol=eps)
            
        # For bounce>=2, triangle inequality: total distance fwd + back >= 2 * back  => r_event >= back_R
        b2_mask = bounces_flat >= 2
        if np.any(b2_mask):
            assert np.all(r_event[b2_mask] >= back_R[b2_mask] - eps)
        
        power_components = pol_valid['Real']**2 + pol_valid['Imag']**2
        power = np.sum(power_components, axis=-1)
        power *= 1.0 / np.maximum(back_R**2, 1e-12)
        
        az_deg = np.degrees(np.arctan2(v_to_target[:, 1], v_to_target[:, 0]))
        
        all_hit_r.append(r_event)
        all_hit_az.append(az_deg)
        all_hit_power.append(power)
        all_hit_bounces.append(bounces_flat)
        
        if return_paths:
            # We must map back the location history arrays for the hits
            # `mask_bounces` is (N_valid, 11). `np.nonzero(mask_bounces)[0]` gives the original row index in `ds_valid`.
            orig_indices = np.nonzero(mask_bounces)[0]
            all_hit_paths.append(locs[orig_indices])
            all_hit_refs.append(valid_refs[orig_indices])
        
    if not all_hit_r:
        if return_paths:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    if return_paths:
        return np.concatenate(all_hit_r), np.concatenate(all_hit_az), np.concatenate(all_hit_power), np.concatenate(all_hit_bounces), np.concatenate(all_hit_paths), np.concatenate(all_hit_refs)
    return np.concatenate(all_hit_r), np.concatenate(all_hit_az), np.concatenate(all_hit_power), np.concatenate(all_hit_bounces)

def generate_ideal_mask(hit_r, hit_az, hit_power, n_angles=1024, angle_min=-80.0, angle_max=80.0,
                        n_range=394, range_min=0.0, range_max=8.0):
    """
    Projects the gridless continuous scattering points onto a discrete mathematical 2D matrix
    acting as an ideal ground truth radar Range-Azimuth map.
    """
    az_edges = np.linspace(angle_min, angle_max, n_angles + 1)
    r_edges = np.linspace(range_min, range_max, n_range + 1)
    
    ra_mask, _, _ = np.histogram2d(
        hit_az, hit_r,
        bins=[az_edges, r_edges],
        weights=hit_power
    )

    # Convert power to dB linearly
    # 10 * log10 applied to power
    ra_mask_db = 10 * np.log10(ra_mask + 1e-12)
    peak_db = np.max(ra_mask_db)
    ra_mask_db = np.clip(ra_mask_db, peak_db - 60, peak_db) # 60dB dynamic floor

    return ra_mask_db

def plot_ideal_mask(ra_mask_db, output_path, title, 
                    n_angles=1024, angle_min=-80.0, angle_max=80.0, 
                    n_range=394, range_min=0.0, range_max=8.0, r_display=8.0):
    """
    Plots the ideal GT Baseband mask using a Polar Matplotlib projection.
    """
    d_r = (range_max - range_min) / n_range
    d_az = (angle_max - angle_min) / n_angles
    
    r_edges_all = range_min + np.arange(n_range + 1) * d_r
    th_edges_all = np.radians(angle_min + np.arange(n_angles + 1) * d_az)
    
    # Filter for display bounds
    r_disp_mask = r_edges_all <= (r_display + d_r)
    r_edges = r_edges_all[r_disp_mask]
    
    # Meshgrid truncation
    n_disp = len(r_edges) - 1
    disp_mask = ra_mask_db[:, :n_disp]

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 8))
    mesh = ax.pcolormesh(th_edges_all, r_edges, disp_mask.T,
                         cmap='magma', shading='flat')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(angle_min)
    ax.set_thetamax(angle_max)
    ax.set_rlabel_position(0)
    ax.set_rticks(np.arange(0, r_display+1, 1))
    ax.set_thetagrids(np.arange(-80, 81, 20))
    ax.grid(True, color='white', alpha=0.3, linewidth=0.5)

    cb = plt.colorbar(mesh, pad=0.1, shrink=0.8)
    cb.set_label('Accumulated Energy (dB)')

    plt.title(title, pad=20)
    plt.savefig(str(output_path), dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()

def plot_multipath_masks(masks_dict, output_path, base_title,
                           n_angles=1024, angle_min=-80.0, angle_max=80.0, 
                           n_range=394, range_min=0.0, range_max=8.0, r_display=8.0):
    """
    Plots multiple ideal GT Baseband masks side by side using Polar Matplotlib projection.
    masks_dict: dict of {"Title": ra_mask_db}
    """
    d_r = (range_max - range_min) / n_range
    d_az = (angle_max - angle_min) / n_angles
    
    r_edges_all = range_min + np.arange(n_range + 1) * d_r
    th_edges_all = np.radians(angle_min + np.arange(n_angles + 1) * d_az)
    
    r_disp_mask = r_edges_all <= (r_display + d_r)
    r_edges = r_edges_all[r_disp_mask]
    n_disp = len(r_edges) - 1

    n_plots = len(masks_dict)
    fig, axes = plt.subplots(1, n_plots, subplot_kw=dict(projection='polar'), figsize=(7 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]
        
    for ax, (title, ra_mask_db) in zip(axes, masks_dict.items()):
        disp_mask = ra_mask_db[:, :n_disp]
        
        mesh = ax.pcolormesh(th_edges_all, r_edges, disp_mask.T,
                             cmap='magma', shading='flat', 
                             vmin=np.max(ra_mask_db)-60, vmax=np.max(ra_mask_db))

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(angle_min)
        ax.set_thetamax(angle_max)
        ax.set_rlabel_position(0)
        ax.set_rticks(np.arange(0, r_display+1, 1))
        ax.set_thetagrids(np.arange(-80, 81, 20))
        ax.grid(True, color='white', alpha=0.3, linewidth=0.5)
        ax.set_title(title, pad=15)

        cb = plt.colorbar(mesh, ax=ax, pad=0.1, shrink=0.6)
        cb.set_label('Accumulated Energy (dB)')

    fig.suptitle(base_title, fontsize=16, y=1.05)
    plt.savefig(str(output_path), dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()

def plot_bounce_distribution(hit_bounces, output_path):
    """
    Bar chart visualization of ray bouncing paths (R-A-R, R-A-B-R, etc.)
    """
    unique_bounces, counts = np.unique(hit_bounces, return_counts=True)
    
    labels = []
    for b in unique_bounces:
        letters = "-".join([chr(65+i) for i in range(b)])
        labels.append(f"{b} Bounce(s)\nR-{letters}-R")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(labels, counts, color='#ff7f0e', edgecolor='white')
    
    ax.set_title("Ray Reflection Pathway Distribution (Ground Truth)", pad=15)
    ax.set_ylabel("Number of Received Hit Points", labelpad=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
