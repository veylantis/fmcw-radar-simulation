"""
radar_viz.py — Visualization library for FMCW radar simulations
══════════════════════════════════════════════════════════
Pure functions: accept numpy data + metadata, save interactive HTML.

Usage:
    import radar_viz as viz
    viz.plot_range_fft(range_axis, profiles_lin, r_markers=[...], output_path=...)

Functions:
    plot_range_fft()                — range profile (all channels + mean)
    plot_range_fft_animated()       — animated range FFT across trajectory snapshots
    plot_range_doppler()            — range-Doppler map (2D heatmap)
    plot_range_doppler_animated()   — animated range-Doppler map (all CPI frames)
    plot_range_angle()              — range-angle map (2D heatmap)
    plot_range_angle_animated()     — animated range-angle map (all CPI frames)
    plot_scene_3d()                 — static 3D scene (radar + targets + trajectories)
    plot_scene_3d_animated()        — animated 3D scene with target moving along trajectory
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

# ── Color palette ─────────────────────────────────────────────────────────────────
_BG_PLOT  = "#0d1117"
_BG_PAPER = "#161b22"
_C_GRID   = "rgba(255,255,255,0.08)"
_C_CHAN   = "rgba(150,160,200,0.30)"    # all channels - grey
_C_HI     = "#00b4d8"                  # highlighted channel - blue
_C_MEAN   = "#f72585"                  # mean - pink
_C_MARKER = "#ffbe0b"                  # target marker - yellow
_FONT     = dict(color="white", family="monospace")

_LAYOUT_BASE = dict(
    plot_bgcolor  = _BG_PLOT,
    paper_bgcolor = _BG_PAPER,
    font          = _FONT,
    legend        = dict(
        bgcolor     = "rgba(0,0,0,0.5)",
        bordercolor = "rgba(255,255,255,0.2)",
        borderwidth = 1,
    ),
    margin = dict(l=60, r=40, t=80, b=50),
)


def _axis(title: str, **kw) -> dict:
    return dict(title=title, color="white", gridcolor=_C_GRID, **kw)


def _save_and_open(fig: go.Figure, output_path: Path | str,
                   open_browser: bool = True, height: int = 520) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(height=height)
    fig.write_html(str(output_path))
    print(f"    ✓ Saved: {output_path}")
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def _build_scene_kw(scene_bounds: dict | None) -> dict:
    """Build 3D scene parameter dict with optional axis bounds."""
    kw = dict(
        xaxis=dict(title="X (m)", backgroundcolor=_BG_PLOT, gridcolor=_C_GRID),
        yaxis=dict(title="Y (m)", backgroundcolor=_BG_PLOT, gridcolor=_C_GRID),
        zaxis=dict(title="Z (m)", backgroundcolor=_BG_PLOT, gridcolor=_C_GRID),
        bgcolor=_BG_PLOT,
    )
    if scene_bounds:
        kw["xaxis"]["range"] = scene_bounds.get("x")
        kw["yaxis"]["range"] = scene_bounds.get("y")
        kw["zaxis"]["range"] = scene_bounds.get("z")
    return kw


# ════════════════════════════════════════════════════════════════════════════
# plot_range_fft
# ════════════════════════════════════════════════════════════════════════════
def plot_range_fft(
    range_axis: np.ndarray,
    profiles_lin: np.ndarray,
    *,
    r_markers: Sequence[tuple[float, str]] = (),
    title: str = "Range FFT",
    output_path: Path | str,
    highlight_ch: int = 0,
    open_browser: bool = True,
) -> go.Figure:
    """
    Plot a range FFT profile.

    Parameters
    ----------
    range_axis   : 1-D array [N_bins]  — range axis in meters
    profiles_lin : 2-D array [n_ch, N_bins] — linear amplitudes per channel
    r_markers    : list of (R_meters, "Label") vertical target markers
    title        : plot title
    output_path  : output path for HTML file
    highlight_ch : channel index to highlight (default: 0)
    open_browser : open browser after saving
    """
    n_ch = profiles_lin.shape[0]
    dB   = lambda x: 20 * np.log10(np.clip(x, 1e-12, None))

    profiles_dB = dB(profiles_lin)
    mean_dB     = dB(np.mean(profiles_lin, axis=0))

    fig = go.Figure()

    # All channels - thin grey lines
    for ch in range(n_ch):
        tx_idx = ch // 4
        rx_idx = ch % 4
        fig.add_trace(go.Scatter(
            x=range_axis, y=profiles_dB[ch],
            name=f"TX{tx_idx+1}-RX{rx_idx+1}",
            line=dict(width=1, color=_C_CHAN),
            showlegend=(ch == 0),
            legendgroup="channels",
            legendgrouptitle=dict(text="All channels") if ch == 0 else None,
            hovertemplate=f"R=%{{x:.3f}} m<br>%{{y:.1f}} dB<extra>TX{tx_idx+1}-RX{rx_idx+1}</extra>",
        ))

    # Highlighted channel
    tx_h = highlight_ch // 4
    rx_h = highlight_ch % 4
    fig.add_trace(go.Scatter(
        x=range_axis, y=profiles_dB[highlight_ch],
        name=f"TX{tx_h+1}–RX{rx_h+1}",
        line=dict(color=_C_HI, width=2),
        hovertemplate=f"R=%{{x:.3f}} m<br>%{{y:.1f}} dB<extra>TX{tx_h+1}-RX{rx_h+1}</extra>",
    ))

    # Mean
    fig.add_trace(go.Scatter(
        x=range_axis, y=mean_dB,
        name=f"Mean ({n_ch} ch)",
        line=dict(color=_C_MEAN, width=2.5),
        hovertemplate="R=%{x:.3f} m<br>%{y:.1f} dB<extra>mean</extra>",
    ))

    # Target markers
    for R, label in r_markers:
        fig.add_vline(
            x=R,
            line=dict(color=_C_MARKER, width=2, dash="dash"),
            annotation_text=label,
            annotation_font_color=_C_MARKER,
            annotation_font_size=12,
            annotation_position="top right",
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color="white")),
        xaxis=_axis("Range (m)", range=[range_axis[0], range_axis[-1]]),
        yaxis=_axis("Amplitude (dB)"),
    )

    return _save_and_open(fig, output_path, open_browser, height=520)


# ════════════════════════════════════════════════════════════════════════════
# plot_range_fft_animated
# ════════════════════════════════════════════════════════════════════════════
def plot_range_fft_animated(
    range_axis: np.ndarray,
    profiles_frames: np.ndarray,
    frame_labels: Sequence[str],
    *,
    r_positions: np.ndarray | None = None,
    title: str = "Range FFT (animated)",
    output_path: Path | str,
    frame_duration_ms: int = 200,
    open_browser: bool = True,
) -> go.Figure:
    """
    Animated range FFT profile across trajectory snapshots.

    Parameters
    ----------
    range_axis      : 1-D array [N_bins] — range axis in meters
    profiles_frames : 3-D array [n_frames, n_ch, N_bins] — linear amplitudes
    frame_labels    : list of strings [n_frames] — label for each frame (slider)
    r_positions     : 1-D array [n_frames] — target positions (m) for marker, or None
    frame_duration_ms : delay between frames during playback (ms)
    """
    n_frames, n_ch, _ = profiles_frames.shape
    dB_fn = lambda x: 20 * np.log10(np.clip(x, 1e-12, None))

    # Y axis: center = signal mean, range = [min, max] + 15% padding
    all_dB    = dB_fn(profiles_frames)           # [n_frames, n_ch, N_bins]
    sig_max   = float(np.max(all_dB))
    sig_min   = float(np.min(all_dB))
    sig_mean  = float(np.mean(all_dB))
    # Distance from center to nearest edge × 1.15 (padding)
    half_range = max(sig_max - sig_mean, sig_mean - sig_min) * 1.15
    y_max = sig_mean + half_range
    y_min = sig_mean - half_range

    HAS_MARKER = r_positions is not None

    def _full_traces(i: int) -> list:
        """Full trace set for frame i (used for initial state)."""
        prof_dB  = dB_fn(profiles_frames[i])            # [n_ch, N_bins]
        mean_dB  = dB_fn(np.mean(profiles_frames[i], axis=0))
        traces = []
        # All channels - grey
        for ch in range(n_ch):
            traces.append(go.Scatter(
                x=range_axis, y=prof_dB[ch],
                mode="lines",
                line=dict(width=0.7, color=_C_CHAN),
                name="All channels",
                legendgroup="channels",
                showlegend=(ch == 0),
                hoverinfo="skip",
            ))
        # Mean
        traces.append(go.Scatter(
            x=range_axis, y=mean_dB,
            mode="lines",
            line=dict(color=_C_MEAN, width=2.5),
            name="Mean",
            hovertemplate="R=%{x:.3f} m<br>%{y:.1f} dB<extra>mean</extra>",
        ))
        # Target marker
        if HAS_MARKER:
            R = float(r_positions[i])
            traces.append(go.Scatter(
                x=[R, R], y=[y_min, y_max],
                mode="lines",
                line=dict(color=_C_MARKER, width=2, dash="dash"),
                name="Target",
                hovertemplate=f"R={R:.2f} m<extra>Target</extra>",
            ))
        return traces

    def _update_traces(i: int) -> list:
        """Data-only (x, y) update for go.Frame."""
        prof_dB = dB_fn(profiles_frames[i])
        mean_dB = dB_fn(np.mean(profiles_frames[i], axis=0))
        upd = []
        for ch in range(n_ch):
            upd.append(go.Scatter(x=range_axis, y=prof_dB[ch]))
        upd.append(go.Scatter(x=range_axis, y=mean_dB))
        if HAS_MARKER:
            R = float(r_positions[i])
            upd.append(go.Scatter(x=[R, R], y=[y_min, y_max]))
        return upd

    # Initial frame
    fig = go.Figure(data=_full_traces(0))

    # Animation frames
    fig.frames = [
        go.Frame(
            data=_update_traces(i),
            name=str(i),
            layout=go.Layout(
                title_text=f"{title}  ·  {frame_labels[i]}"
            ),
        )
        for i in range(n_frames)
    ]

    # Slider
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=frame_duration_ms, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{i:02d}",
            method="animate",
        )
        for i in range(n_frames)
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"{title}  ·  {frame_labels[0]}",
            font=dict(size=13, color="white"),
            y=0.98, x=0.5, xanchor="center", yanchor="top",
        ),
        xaxis=_axis("Range (m)", range=[range_axis[0], range_axis[-1]]),
        yaxis=_axis("Amplitude (dB)", range=[y_min, y_max]),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.22, x=0.0, xanchor="left", yanchor="top",
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True, mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            currentvalue=dict(
                prefix="Frame: ", visible=True, xanchor="center",
                font=dict(color="white", size=12),
            ),
            pad=dict(b=10, t=40),
            len=0.9, x=0.05,
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
        )],
    )
    # Increase bottom margin for slider + buttons
    fig.update_layout(margin=dict(l=60, r=40, t=50, b=120))

    return _save_and_open(fig, output_path, open_browser, height=820)





# ════════════════════════════════════════════════════════════════════════════
# plot_range_doppler
# ════════════════════════════════════════════════════════════════════════════
def plot_range_doppler(
    rd_map: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    *,
    title: str = "Range-Doppler",
    output_path: Path | str,
    db_floor: float = -60.0,
    open_browser: bool = True,
) -> go.Figure:
    """
    Plot a range-Doppler map (2D heatmap).

    Parameters
    ----------
    rd_map        : 2-D array [n_doppler, n_range] — linear amplitude
    range_axis    : 1-D array [n_range]    — range axis, m
    velocity_axis : 1-D array [n_doppler]  — velocity axis, m/s
    db_floor      : display threshold (dB below peak)
    """
    rd_dB = 20 * np.log10(np.clip(rd_map, 1e-12, None))
    rd_dB = np.clip(rd_dB, rd_dB.max() + db_floor, None)

    fig = go.Figure(go.Heatmap(
        z=rd_dB,
        x=range_axis,
        y=velocity_axis,
        colorscale="Plasma",
        colorbar=dict(title="dB", tickfont=dict(color="white")),
        hovertemplate="R=%{x:.2f} m<br>v=%{y:.2f} m/s<br>%{z:.1f} dB<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color="white")),
        xaxis=_axis("Range (m)"),
        yaxis=_axis("Velocity (m/s)"),
    )

    return _save_and_open(fig, output_path, open_browser, height=540)


# ════════════════════════════════════════════════════════════════════════════
# plot_range_doppler_animated
# ════════════════════════════════════════════════════════════════════════════
def plot_range_doppler_animated(
    rd_maps: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    frame_labels: Sequence[str],
    *,
    title: str = "Range-Doppler (animated)",
    output_path: Path | str,
    db_floor: float = -60.0,
    frame_duration_ms: int = 300,
    open_browser: bool = True,
) -> go.Figure:
    """
    Animated range-Doppler map with one frame per CPI.

    Parameters
    ----------
    rd_maps       : 3-D array [n_frames, n_doppler, n_range] — linear amplitude
    range_axis    : 1-D array [n_range]    — range axis, m
    velocity_axis : 1-D array [n_doppler]  — velocity axis, m/s
    frame_labels  : list of strings [n_frames] — frame labels (slider and title)
    db_floor      : display threshold (dB below global peak)
    frame_duration_ms : delay between frames during playback (ms)

    Notes
    -----
    zmin/zmax computed globally across all frames for stable color scale.
    Physical effects are visible: peak amplitude changes are preserved.
    """
    n_frames = rd_maps.shape[0]

    # Convert to dB with global normalization
    rd_dB_all = 20 * np.log10(np.clip(rd_maps, 1e-12, None))  # [n_frames, n_dop, n_range]
    z_max = float(rd_dB_all.max())
    z_min = z_max + db_floor

    def _z(i: int) -> np.ndarray:
        return np.clip(rd_dB_all[i], z_min, None)

    # Initial frame
    fig = go.Figure(go.Heatmap(
        z=_z(0),
        x=range_axis,
        y=velocity_axis,
        colorscale="Plasma",
        zmin=z_min, zmax=z_max,
        colorbar=dict(
            title=dict(text="dB", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
        hovertemplate="R=%{x:.2f} m<br>v=%{y:.3f} m/s<br>%{z:.1f} dB<extra></extra>",
    ))

    # Animation frames
    fig.frames = [
        go.Frame(
            data=[go.Heatmap(z=_z(i))],
            name=str(i),
            layout=go.Layout(title_text=f"{title}  ·  {frame_labels[i]}"),
        )
        for i in range(n_frames)
    ]

    # Slider
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=frame_duration_ms, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{i:02d}",
            method="animate",
        )
        for i in range(n_frames)
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"{title}  ·  {frame_labels[0]}",
            font=dict(size=13, color="white"),
        ),
        xaxis=_axis("Range (m)"),
        yaxis=_axis("Velocity (m/s)"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.18, x=0.0, xanchor="left",
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True, mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            currentvalue=dict(
                prefix="Frame: ", visible=True, xanchor="center",
                font=dict(color="white", size=12),
            ),
            pad=dict(b=10, t=60),
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
        )],
    )

    return _save_and_open(fig, output_path, open_browser, height=580)


# ════════════════════════════════════════════════════════════════════════════
# plot_range_angle
# ════════════════════════════════════════════════════════════════════════════
def plot_range_angle(
    ra_map: np.ndarray,
    range_axis: np.ndarray,
    angle_axis: np.ndarray,
    *,
    title: str = "Range-Angle",
    output_path: Path | str,
    db_floor: float = -60.0,
    open_browser: bool = True,
) -> go.Figure:
    """
    Plot a range-angle map (2D heatmap).

    Parameters
    ----------
    ra_map      : 2-D array [n_angle, n_range] — linear amplitude
    range_axis  : 1-D array [n_range]  — range axis, m
    angle_axis  : 1-D array [n_angle]  — azimuth axis, degrees
    """
    ra_dB = 20 * np.log10(np.clip(ra_map, 1e-12, None))
    ra_dB = np.clip(ra_dB, ra_dB.max() + db_floor, None)

    fig = go.Figure(go.Heatmap(
        z=ra_dB,
        x=range_axis,
        y=angle_axis,
        colorscale="Viridis",
        colorbar=dict(title="dB", tickfont=dict(color="white")),
        hovertemplate="R=%{x:.2f} m<br>Az=%{y:.1f}°<br>%{z:.1f} dB<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color="white")),
        xaxis=_axis("Range (m)"),
        yaxis=_axis("Azimuth (deg)"),
    )

    return _save_and_open(fig, output_path, open_browser, height=540)


# ════════════════════════════════════════════════════════════════════════════
# plot_range_angle_animated
# ════════════════════════════════════════════════════════════════════════════
def plot_range_angle_animated(
    ra_maps: np.ndarray,
    range_axis: np.ndarray,
    angle_axis: np.ndarray,
    frame_labels: Sequence[str],
    *,
    title: str = "Range-Angle (animated)",
    output_path: Path | str,
    db_floor: float = -60.0,
    frame_duration_ms: int = 300,
    open_browser: bool = True,
) -> go.Figure:
    """
    Animated range-angle map with one frame per CPI.

    Parameters
    ----------
    ra_maps       : 3-D array [n_frames, n_angles, n_range] — values in dB
    range_axis    : 1-D [n_range]  — range axis, m
    angle_axis    : 1-D [n_angles] — azimuth axis, degrees
    frame_labels  : [n_frames] — frame labels
    db_floor      : threshold relative to global peak

    Notes
    -----
    ra_maps may already be in dB (e.g. from doa_capon) or linear amplitude.
    Scale range is determined globally.
    """
    n_frames = ra_maps.shape[0]

    # Global normalization
    z_max = float(ra_maps.max())
    z_min = z_max + db_floor

    def _z(i: int) -> np.ndarray:
        return np.clip(ra_maps[i], z_min, None)

    # Initial frame
    fig = go.Figure(go.Heatmap(
        z=_z(0),
        x=range_axis,
        y=angle_axis,
        colorscale="Viridis",
        zmin=z_min, zmax=z_max,
        colorbar=dict(
            title=dict(text="dB", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
        hovertemplate="R=%{x:.2f} m<br>Az=%{y:.1f}°<br>%{z:.1f} dB<extra></extra>",
    ))

    # Animation frames
    fig.frames = [
        go.Frame(
            data=[go.Heatmap(z=_z(i))],
            name=str(i),
            layout=go.Layout(title_text=f"{title}  ·  {frame_labels[i]}"),
        )
        for i in range(n_frames)
    ]

    # Slider
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=frame_duration_ms, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{i:02d}",
            method="animate",
        )
        for i in range(n_frames)
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"{title}  ·  {frame_labels[0]}",
            font=dict(size=13, color="white"),
        ),
        xaxis=_axis("Range (m)"),
        yaxis=_axis("Azimuth (deg)"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.18, x=0.0, xanchor="left",
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True, mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            currentvalue=dict(
                prefix="Frame: ", visible=True, xanchor="center",
                font=dict(color="white", size=12),
            ),
            pad=dict(b=10, t=60),
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
        )],
    )

    return _save_and_open(fig, output_path, open_browser, height=580)


# ════════════════════════════════════════════════════════════════════════════
# plot_scene_3d
# ════════════════════════════════════════════════════════════════════════════
def plot_scene_3d(
    targets: Sequence[dict],
    *,
    radar_pos: tuple = (0.0, 0.0, 0.0),
    title: str = "3D Scene",
    output_path: Path | str,
    open_browser: bool = True,
    trajectories: Sequence[dict] | None = None,
    scene_bounds: dict | None = None,
) -> go.Figure:
    """
    Plot a static 3D scene: radar + targets + trajectories.

    Parameters
    ----------
    targets      : list of dicts {"location": (x,y,z), "label": str, "rcs_dBsm": float}
    radar_pos    : radar position (x, y, z)
    trajectories : list of dicts {
                       "xyz": array [N, 3],       — trajectory points
                       "label": str,              — legend label
                       "color": str (optional),   — line color
                   }
    scene_bounds : {"x": [min, max], "y": [min, max], "z": [min, max]}
                   Must be set for correct scene rendering.
                   Add ~2 m padding around the farthest object.
    """
    fig = go.Figure()

    # Radar
    fig.add_trace(go.Scatter3d(
        x=[radar_pos[0]], y=[radar_pos[1]], z=[radar_pos[2]],
        mode="markers+text",
        marker=dict(size=8, color=_C_HI, symbol="diamond"),
        text=["Radar"], textposition="top center",
        name="Radar",
    ))

    # Trajectories
    if trajectories:
        for traj in trajectories:
            xyz   = np.asarray(traj["xyz"])
            lbl   = traj.get("label", "Trajectory")
            color = traj.get("color", "rgba(100,200,100,0.7)")
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines",
                line=dict(color=color, width=4),
                name=lbl,
                hovertemplate=(
                    "x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m"
                    "<extra>" + lbl + "</extra>"
                ),
            ))
            # Start/end labels
            for pt, txt in [(xyz[0], "Start"), (xyz[-1], "End")]:
                fig.add_trace(go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode="markers+text",
                    marker=dict(size=5, color=color),
                    text=[txt], textposition="top center",
                    showlegend=False,
                ))

    # Targets (initial position)
    for t in targets:
        loc = t.get("location", (0, 0, 0))
        lbl = t.get("label", "Target")
        rcs = t.get("rcs_dBsm", None)
        hover = f"{lbl}<br>({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) m"
        if rcs is not None:
            hover += f"<br>RCS={rcs:.1f} dBsm"
        fig.add_trace(go.Scatter3d(
            x=[loc[0]], y=[loc[1]], z=[loc[2]],
            mode="markers+text",
            marker=dict(size=7, color=_C_MARKER),
            text=[lbl], textposition="top center",
            name=lbl,
            hovertemplate=hover + "<extra></extra>",
        ))
        # Radar-to-target line
        fig.add_trace(go.Scatter3d(
            x=[radar_pos[0], loc[0]],
            y=[radar_pos[1], loc[1]],
            z=[radar_pos[2], loc[2]],
            mode="lines",
            line=dict(color="rgba(255,190,11,0.3)", width=2),
            showlegend=False,
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color="white")),
        scene=_build_scene_kw(scene_bounds),
    )

    return _save_and_open(fig, output_path, open_browser, height=620)


# ════════════════════════════════════════════════════════════════════════════
# plot_scene_3d_animated
# ════════════════════════════════════════════════════════════════════════════
def plot_scene_3d_animated(
    trajectory_xyz: np.ndarray,
    frame_labels: Sequence[str],
    *,
    radar_pos: tuple = (0.0, 0.0, 0.0),
    target_label: str = "Target",
    rcs_dBsm: float | None = None,
    title: str = "3D Scene (animated)",
    output_path: Path | str,
    open_browser: bool = True,
    frame_duration_ms: int = 120,
    scene_bounds: dict | None = None,
) -> go.Figure:
    """
    Animated 3D scene: target moves point-by-point along a trajectory.

    Parameters
    ----------
    trajectory_xyz  : array [N_frames, 3] - target position at each frame
    frame_labels    : list of strings [N_frames] - frame labels (slider and title)
    radar_pos       : radar position (x, y, z)
    target_label    : target name in legend
    rcs_dBsm        : target RCS (dBsm), shown in hover tooltip
    frame_duration_ms : delay between frames during playback (ms)
    scene_bounds    : {"x": [min, max], "y": [min, max], "z": [min, max]}
                      Must be set, otherwise the camera jumps during animation.

    Notes
    -----
    Frame architecture:
      - Traces 0,1,2 are static (radar, trajectory line, Start/End labels)
      - Traces 3,4 are dynamic (target position + radar-to-target beam)
      - go.Frame(traces=[3,4]) updates only the dynamic traces
    """
    xyz = np.asarray(trajectory_xyz)    # [N_frames, 3]
    n_frames = len(xyz)

    rcs_str = f"<br>RCS={rcs_dBsm:.1f} dBsm" if rcs_dBsm is not None else ""

    # ── Static traces ─────────────────────────────────────────────────────
    t_radar = go.Scatter3d(
        x=[radar_pos[0]], y=[radar_pos[1]], z=[radar_pos[2]],
        mode="markers+text",
        marker=dict(size=8, color=_C_HI, symbol="diamond"),
        text=["Radar"], textposition="top center",
        name="Radar",
    )
    t_traj = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="lines",
        line=dict(color="rgba(100,220,100,0.45)", width=3),
        name="Trajectory",
        hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra>trajectory</extra>",
    )
    t_endpoints = go.Scatter3d(
        x=[xyz[0, 0], xyz[-1, 0]],
        y=[xyz[0, 1], xyz[-1, 1]],
        z=[xyz[0, 2], xyz[-1, 2]],
        mode="markers+text",
        marker=dict(
            size=6,
            color=["rgba(100,220,100,0.9)", "rgba(220,80,80,0.9)"],
        ),
        text=["Start", "End"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    )

    # ── Dynamic traces (updated each frame) ──────────────────────
    def _target_trace(i: int) -> go.Scatter3d:
        x, y, z = xyz[i]
        hover = f"{target_label}<br>({x:.2f}, {y:.2f}, {z:.2f}) m{rcs_str}"
        return go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text",
            marker=dict(
                size=10, color=_C_MARKER,
                line=dict(color="white", width=1.5),
            ),
            text=[target_label],
            textposition="top center",
            name=target_label,
            hovertemplate=hover + "<extra></extra>",
        )

    def _beam_trace(i: int) -> go.Scatter3d:
        x, y, z = xyz[i]
        return go.Scatter3d(
            x=[radar_pos[0], x],
            y=[radar_pos[1], y],
            z=[radar_pos[2], z],
            mode="lines",
            line=dict(color="rgba(255,190,11,0.3)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )

    # Initial state: idx=0,1,2 static; idx=3,4 dynamic
    fig = go.Figure(data=[
        t_radar,             # idx=0
        t_traj,              # idx=1
        t_endpoints,         # idx=2
        _target_trace(0),    # idx=3  ─┐ updated
        _beam_trace(0),      # idx=4  ─┘ per frame
    ])

    # ── Animation frames ───────────────────────────────────────────────────
    fig.frames = [
        go.Frame(
            data=[_target_trace(i), _beam_trace(i)],
            traces=[3, 4],
            name=str(i),
            layout=go.Layout(title_text=f"{title}  ·  {frame_labels[i]}"),
        )
        for i in range(n_frames)
    ]

    # ── Slider ──────────────────────────────────────────────────────────────
    slider_steps = [
        dict(
            args=[[str(i)], dict(
                frame=dict(duration=frame_duration_ms, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            )],
            label=f"{i:02d}",
            method="animate",
        )
        for i in range(n_frames)
    ]

    # ── Layout assembly ─────────────────────────────────────────────────────────
    scene_kw = _build_scene_kw(scene_bounds)
    scene_kw["aspectmode"] = "manual"
    scene_kw["aspectratio"] = dict(x=1.5, y=1.5, z=0.4)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f"{title}  ·  {frame_labels[0]}",
            font=dict(size=13, color="white"),
        ),
        scene=scene_kw,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12, x=0.0, xanchor="left",
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True, mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            currentvalue=dict(
                prefix="Frame: ", visible=True, xanchor="center",
                font=dict(color="white", size=12),
            ),
            pad=dict(b=10, t=55),
            bgcolor=_BG_PAPER,
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
        )],
    )

    return _save_and_open(fig, output_path, open_browser, height=700)
