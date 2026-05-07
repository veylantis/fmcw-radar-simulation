# Radar Configuration — TI xWR6843AOP

> This file is the **reference hardware config** used across all simulations.  
> Modify only when changing hardware or performing a deliberate recalibration.  
> Source: TI mmWave Sensing Estimator + SWRA656 + SWRS237.

---

## Device

| Parameter | Value |
|-----------|-------|
| Chip | **TI xWR6843AOP** (IWR6843AOP ES2.0) |
| Type | Antenna-on-Package (integrated antennas) |
| Frequency Band | 60–64 GHz |
| TX Antennas | 3 |
| RX Antennas | 4 |
| Virtual Channels | **12** (3 TX × 4 RX) |
| TX Gain | 5.2 dB |
| RX Gain | 5.2 dB |
| TX Power | 12 dBm |

---

## Chirp Parameters (Short Range Default)

> Source: TI mmWave Sensing Estimator, "Short Range Default" profile

| Parameter | Value | Unit |
|-----------|-------|------|
| Frequency Start | 60 | GHz |
| Frequency Slope | 22.788 | MHz/µs |
| Ramp End Time | 138.733 | µs |
| ADC Valid Start Time | 6.4 | µs |
| Idle Time | 7 | µs |
| Samples per Chirp | 394 | — |
| Sample Rate (ADC) | 3000 | ksps |
| Chirp Loops | 2 | — |
| Chirps in Burst | 4 | — |
| Frame Periodicity | 50 | ms |

### Derived Parameters

| Parameter | Value | Formula |
|-----------|-------|---------|
| Effective Bandwidth | ~3161 MHz | Slope × ADC time |
| Max Bandwidth | 4000 MHz | — |
| Range Resolution | **~4.7 cm** | c / (2B) |
| Range Interbin | 3.848 cm | — |
| Max Range (typical) | 32.1 m | from loss params |
| Beat Frequency | 1.520 MHz | — |
| Chirp Cycle Time | 145.733 µs | — |
| Velocity Resolution | 5.556 m/s | — |
| Range FFT Bins | 512 | — |
| Doppler FFT Bins | 2 | — |
| Min RCS @ max range | 0.009 m² | Adult ~1 m² |

### profileCfg (raw TI config)

```
profileCfg 0 60 7 6.4 138.73 0 0 22.79 1 394 3000 0 0 30
channelCfg 15 7 0
adcbufCfg -1 0 1 1 1
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 2 0 50 1 0
```

---

## Antenna Geometry (ES2.0, 2D MIMO)

> Source: TI SWRA656, SWRS237 §7.12.2  
> Units: multiples of λ/2, λ ≈ 2.5 mm at 60 GHz

### TX Physical Positions (in λ/2 units)

| Antenna | Azimuth (Y) | Elevation (Z) |
|---------|-------------|----------------|
| TX1 | 0 | 0 |
| TX2 | 4 | 0 |
| TX3 | 0 | 2 |

### RX Physical Positions (in λ/2 units)

| Antenna | Azimuth (Y) | Elevation (Z) |
|---------|-------------|----------------|
| RX1 | 0 | 0 |
| RX2 | 1 | 0 |
| RX3 | 2 | 0 |
| RX4 | 3 | 0 |

### Virtual Array (12 channels, v = TX + RX)

| TX | RX | Virtual Position (Y, Z) |
|----|----|--------------------------|
| TX1 | RX1–4 | (0,0), (1,0), (2,0), (3,0) |
| TX2 | RX1–4 | (4,0), (5,0), (6,0), (7,0) |
| TX3 | RX1–4 | (0,2), (1,2), (2,2), (3,2) |

**Total**: 8 azimuth channels + 2 elevation rows → **2D angular resolution**

### ES2.0 Notes

- RX1↔RX2 and RX3↔RX4 are physically swapped (vs ES1.0)
- TX2↔TX3 are swapped (vs ES1.0)
- When using RadarSimPy, positions are specified explicitly — the swap has no effect

---

## Conversion to Meters (for RadarSimPy)

```python
import numpy as np

fc = 60.5e9          # center frequency (midpoint of 60-64 GHz)
c  = 3e8
lam = c / fc         # ≈ 4.96 mm
d  = lam / 2         # array spacing ≈ 2.48 mm

# TX positions [x, y, z] in meters
tx_locations = [
    [0, 0 * d, 0 * d],   # TX1
    [0, 4 * d, 0 * d],   # TX2
    [0, 0 * d, 2 * d],   # TX3
]

# RX positions [x, y, z] in meters
rx_locations = [
    [0, 0 * d, 0],   # RX1
    [0, 1 * d, 0],   # RX2
    [0, 2 * d, 0],   # RX3
    [0, 3 * d, 0],   # RX4
]
```

---

## TDM Scheme (Time-Division Multiplexing)

Each TX transmits in a separate time slot via the `delay` parameter:

| TX | Delay | Slot |
|----|-------|------|
| TX1 | 0 | 0–138 µs |
| TX2 | 145.7 µs | 145–284 µs |
| TX3 | 291.5 µs | 292–430 µs |

```
PRP ≥ 3 × chirp_cycle = 3 × 145.7 µs ≈ 437 µs
```

---

## Target Application Scenarios (from TI Estimator)

| Parameter | Value |
|-----------|-------|
| Max Range | 32 m |
| Range Resolution | 5 cm |
| Max Velocity | 10 km/h |
| Typical Target | Human (1 m²) |
| Update Rate | 20 Hz |
| Application | Short-range: office, perimeter, gestures |
