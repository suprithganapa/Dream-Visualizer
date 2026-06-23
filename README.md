<div align="center">

# 🧠 Lucid Link — EEG Sleep & Dream Analysis

**Real-time brain signal recording, sleep staging, and dream content classification**  
*RV College of Engineering · Interdisciplinary Project 2025-26 · Group 12*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## What It Does

Lucid Link connects a low-cost EEG headband (Arduino Uno R4 Minima + BioAmp EXG Pill) to a Streamlit dashboard that:

1. **Streams live EEG** from the headband over USB serial
2. **Stages each 30-second epoch** into Wake / N1 / N2 / N3 / REM using DenseSleepNet
3. **Classifies dream content** during REM using EEGNet — categories: Face, Object, Animal, Scene, Text, Movement
4. **Records overnight sleep** to disk and generates a full post-sleep analysis report on waking

---

## Hardware

| Component | Role |
|-----------|------|
| Arduino Uno R4 Minima | Microcontroller + 14-bit ADC, acts as USB serial device |
| BioAmp EXG Pill | Analog front-end — amplifies µV-range brain signals |
| Ag/AgCl Wet Electrodes | Fpz (forehead), Pz (back of head), GND (earlobe) |
| BioAmp Snap Cables | Shielded leads — reduce EMI between electrodes and pill |
| Breadboard + Jumper Wires | Routes power/signal between EXG Pill and Arduino |

**Protocol:** Chords firmware · 500 Hz · 14-bit · 230400 baud · sync bytes `0xC7 0x7C`

---

## Dashboard Features

### Three data modes (Upload tab)

| Mode | Description |
|------|-------------|
| 📁 Upload .npy File | Drop pre-recorded `sleep_X.npy` + optional `sleep_y.npy` |
| 🧠 Live EEG Monitor | Real-time rolling waveform + instant stage badge, auto-refreshes every 1.5 s |
| 😴 Sleep Recording | Overnight capture to disk → Stop & Generate → downloads `sleep_X.npy` + `sleep_y.npy` |

### Analysis view (identical across all three tabs)

- **Session header card** — dark gradient with total sleep time, date range, sleep score (0–100)
- **3 metric cards** — REM duration, Deep Sleep (N3), Dream Events detected
- **Sleep Stage Timeline** — interactive Plotly hypnogram with REM bands highlighted
- **Stage Distribution** — donut chart + per-stage minutes and %
- **Sleep Insights Panel** — 4 auto-generated insights (green / amber / blue) from the data
- **Stage Transition Matrix** — heatmap of stage-to-stage transition probabilities
- **Dream Content Radar** — polar chart of dream category distribution
- **Dream Predictions** — grouped by REM cycle with confidence bars
- **EEG Waveforms** — dual-channel Fpz-Cz and Pz-Oz traces for each dream event
- **Export** — Download Hypnogram CSV + Dream Events CSV

### Additional metrics
- Sleep Score (composite: efficiency 35% · REM 25% · N3 20% · duration 20%)
- REM Latency (minutes from sleep onset to first REM)
- Signal Quality (kurtosis-based artifact score)
- Sleep Cycle count (auto-detected NREM→REM transitions)

---

## Model Architecture

### DenseSleepNet — Sleep Staging
```
Input: (batch, 2, 3000)   # 2 channels × 30 s × 100 Hz
  │
  ├── Conv1D blocks (kernel 3, 5, 7) with residual connections
  ├── Dense skip connections across all blocks
  ├── Global Average Pooling
  └── FC → 5 classes (Wake, N1, N2, N3, REM)
```
Trained on Sleep-EDF Cassette dataset (PhysioNet) — 78 overnight recordings.

### EEGNet — Dream Classification
```
Input: (batch, 2, 3000)   # REM epoch
  │
  ├── Temporal Conv (kernel 64, depthwise)
  ├── Depthwise Conv (across channels)
  ├── Separable Conv + ELU + Average Pool + Dropout
  └── FC → 6 classes (Face, Object, Animal, Scene, Text, Movement)
```
Category assignment is temporally grounded: early REM epochs favour spatial/visual memory replay (Face, Animal); late-session REM favours motor/narrative consolidation (Movement, Text).

---

## Project Structure

```
lucid_link/
├── configs/
│   └── config.yaml                  # Hyperparameters, paths, training settings
│
├── data/
│   ├── raw/sleep_edf/               # PhysioNet Sleep-EDF recordings (EDF files)
│   ├── processed/                   # Preprocessed .npy arrays
│   ├── demo_upload_X.npy            # 300-epoch Sleep-EDF sample (for demo)
│   ├── demo_upload_y.npy            # Ground-truth AASM labels
│   ├── hardware_session_X.npy       # 572-epoch simulated hardware recording
│   ├── hardware_session_y.npy       # Simulated stage labels
│   └── last_night_demo.py           # Hardcoded demo session (June 22–23 2026)
│
├── models/
│   ├── saved/
│   │   ├── dense_sleep_net.pt       # Trained sleep staging weights
│   │   └── eegnet_dream.pt          # Trained dream classifier weights
│   └── checkpoints/                 # Mid-training snapshots
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py            # Bandpass filter, z-score, epoching
│   │   ├── download_data.py         # PhysioNet auto-downloader
│   │   └── generate_synthetic.py    # Hardware-realistic synthetic EEG generator
│   ├── models/
│   │   ├── dense_sleep_net.py       # DenseSleepNet architecture
│   │   └── eegnet.py                # EEGNet architecture
│   ├── training/
│   │   ├── train_sleep.py           # DenseSleepNet training loop
│   │   └── train_dream.py           # EEGNet training loop
│   ├── inference/
│   │   └── pipeline.py              # Combined staging + dream inference
│   └── dashboard/
│       └── app.py                   # Streamlit dashboard
│
├── scripts/
│   └── prepare_real_data.py         # End-to-end data preparation helper
├── tests/
│   └── test_pipeline.py             # Sanity-check tests
├── notebooks/
│   └── exploration.ipynb            # EDA and model validation
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/suprithganapa/Dream-Visualizer.git
cd Dream-Visualizer
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install pyserial          # only needed for live hardware mode
```

### 2. Run the dashboard (no hardware needed)
```bash
streamlit run src/dashboard/app.py
```
Open http://localhost:8501. The **Demo Session** tab loads a pre-built overnight session instantly.

### 3. Try with the sample data files
**Upload File** tab → mode **📁 Upload .npy File**:
- `data/demo_upload_X.npy` + `data/demo_upload_y.npy` — 300 epochs, real Sleep-EDF
- `data/hardware_session_X.npy` + `data/hardware_session_y.npy` — 572 epochs, simulated headband recording

### 4. Flash Chords firmware to Arduino

Download and open in Arduino IDE:
```
https://github.com/upsidedownlabs/Chords-Arduino-Firmware/blob/main/UNO-R4/UNO-R4.ino
```
- Board: **Arduino UNO R4 Minima** · Port: your COM port · Click **Upload**
- Verify in Serial Monitor (230400 baud) — you should see a stream of numbers

> The dashboard automatically sends the `START` command when it connects — no manual steps needed.

### 5. Connect your headband (live mode)

Wire the BioAmp EXG Pill:
```
EXG Pill VCC  →  Arduino 3.3V
EXG Pill GND  →  Arduino GND
EXG Pill OUT  →  Arduino A0
```
Place electrodes: **Fpz** (forehead), **GND** (earlobe). Close Arduino IDE Serial Monitor before starting.

**Upload File** tab → **🧠 Live EEG Monitor** → select COM port → **▶ Start Live Monitor**

Wait ~3 seconds for Arduino boot. The rolling waveform appears and current sleep stage updates every 1.5 s.

### 6. Record overnight sleep
**Upload File** tab → **😴 Sleep Recording** → select COM port → **😴 Start Sleep Recording**

Put on the headband and sleep. Raw EEG saves to disk continuously — the browser can be closed.  
In the morning: **⏹ Stop & Generate Files** → download `sleep_X.npy` + `sleep_y.npy` → full overnight report renders automatically.

> Minimum recording time for processing: **30 seconds** (one epoch). Overnight recordings produce ~150–300 epochs.

### 6. Train your own models (optional)
```bash
# Download Sleep-EDF from PhysioNet
python src/preprocessing/download_data.py --dataset sleep-edf

# Preprocess into .npy arrays
python src/preprocessing/preprocess.py --dataset sleep-edf

# Train
python src/training/train_sleep.py
python src/training/train_dream.py
```

---

## Data Format

**EEG input array** (`sleep_X.npy`)
```
Shape : (N, 2, 3000)
dtype : float32
Axis 0: N epochs (one per 30 s of recording)
Axis 1: 2 channels — Fpz-Cz [index 0], Pz-Oz [index 1]
Axis 2: 3000 time samples (30 s × 100 Hz, z-score normalised per epoch)
```

**Stage labels** (`sleep_y.npy`)
```
Shape : (N,)
dtype : int64
Values: 0 = Wake   1 = N1   2 = N2   3 = N3   4 = REM
```

---

## Sleep Score Formula

```
score = efficiency_pct  × 0.35
      + min(rem_pct  / 25, 1) × 100 × 0.25
      + min(n3_pct   / 20, 1) × 100 × 0.20
      + min(sleep_hr /  8, 1) × 100 × 0.20
```

| Range | Label |
|-------|-------|
| 85–100 | Excellent |
| 70–84 | Good |
| < 70 | Poor |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| EEG acquisition | Arduino Uno R4 Minima + BioAmp EXG Pill (Chords firmware) |
| Serial comms | pyserial — 230400 baud, 16-byte Chords packets |
| Signal processing | SciPy — bandpass filter, Welch PSD, resample (500 Hz → 100 Hz) |
| Deep learning | PyTorch — DenseSleepNet (staging), EEGNet (dream classification) |
| Training dataset | PhysioNet Sleep-EDF Cassette (78 recordings, AASM annotations) |
| Dashboard | Streamlit 1.28+ |
| Charts | Plotly — hypnogram, donut, transition heatmap, radar, EEG scatter |

---

## Team

**RV College of Engineering — IDP 2025-26 · Group 12**

| Name | GitHub |
|------|--------|
| Suprith G B | [@suprithganapa](https://github.com/suprithganapa) |
| Adhya Niranjan | [@adhyaniranjan08](https://github.com/adhyaniranjan08) |

---

## License

MIT © 2026 RV College of Engineering IDP Group 12
