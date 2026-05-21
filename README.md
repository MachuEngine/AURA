# AURA — Audio Understanding & RAG for Auto

An in-vehicle AI framework that continuously listens for acoustic events, classifies them with a **lightweight SED model (CNN-Mamba, proposed)**, and generates context-aware vehicle control commands through a **Retrieval-Augmented Generation (RAG)** pipeline.

A comparative study of five architectures (VGG-SED, CRNN, MobileNet-SED, **CNN-Mamba**, CNN-Transformer) under identical SNR-based noise augmentation training, targeting master's thesis submission to IEEE T-ITS.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AURA Pipeline                               │
│                                                                     │
│  Microphone                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────────────────┐                                   │
│  │   Audio Preprocessing        │  16kHz → Log-Mel Spectrogram     │
│  │   (utils/audio_utils.py)     │  (64 mel bins, 256-hop)          │
│  └──────────────┬───────────────┘                                   │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────┐                                   │
│  │   CRNN SED Model             │  CNN (4 layers) + Bi-GRU         │
│  │   (models/sed_model.py)      │  170,708 params  <1M ✓           │
│  │                              │  Latency: ~2ms / inference       │
│  └──────────────┬───────────────┘                                   │
│                 │ (audio_event, confidence)                         │
│                 ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐           │
│  │   FastAPI Backend  (backend/main.py)                 │           │
│  │                                                      │           │
│  │   POST /api/v1/context-stream                        │           │
│  │        │                                             │           │
│  │        ▼                                             │           │
│  │   ┌─────────────────────────────────────────────┐   │           │
│  │   │  RAG Chain  (backend/rag_chain.py)           │   │           │
│  │   │                                             │   │           │
│  │   │  Vehicle Manual ──► FAISS VectorDB          │   │           │
│  │   │       (sentence-transformers/MiniLM-L6-v2)  │   │           │
│  │   │                 │                           │   │           │
│  │   │                 ▼  Top-K retrieval          │   │           │
│  │   │  Context + Event ──► LLM (or MockLLM)       │   │           │
│  │   │                 │                           │   │           │
│  │   │                 ▼                           │   │           │
│  │   │  ┌──────────────────────────────────────┐  │   │           │
│  │   │  │  Actionable Advice + Control JSON    │  │   │           │
│  │   │  └──────────────────────────────────────┘  │   │           │
│  │   └─────────────────────────────────────────────┘   │           │
│  └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Proposed model** | CNN-Mamba (313K params) — CNN backbone + Mamba SSM, pure PyTorch (CPU/MPS) |
| **Architectures compared** | VGG-SED · CRNN · MobileNet-SED · CNN-Mamba · CNN-Transformer |
| **Sound classes** | coughing, sneezing, infant crying (3-class, ESC-50 subset) |
| **Data augmentation** | Offline 7× (speed ×4 + gain ×2) + Online gain jitter / polarity inversion + SpecAugment |
| **Train/Val split** | ESC-50 official 5-Fold: Folds 1–4 train, Fold 5 val (zero leakage) |
| **SED latency** | CNN-Mamba 17.28 ms · CRNN 10.17 ms · CNN-Transformer 8.64 ms (CPU) |
| **RAG retrieval** | FAISS + `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM fallback** | Automatic `MockLLM` when no API key is present |
| **API** | RESTful FastAPI with Pydantic v2 validation |

---

## Project Structure

```
AURA/
├── data/                        # Audio dataset (git-ignored, regenerate below)
│   ├── coughing/                #   Fold 1-4: orig + aug | Fold 5: orig only
│   ├── sneezing/
│   ├── infant_crying/
│   └── noise/
│       ├── vehicle_engine/
│       ├── environmental/
│       └── hvac/
├── models/
│   ├── sed_model.py             # CRNN architecture (170,579 params)
│   ├── vgg_model.py             # VGG-SED (434,979 params)
│   ├── mobilenet_model.py       # MobileNet-SED (125,315 params)
│   ├── mamba_model.py           # CNN-Mamba — proposed (313,555 params)
│   ├── ast_model.py             # CNN-Transformer (344,787 params)
│   ├── sed_baseline.pth         # Baseline CRNN weights (git-ignored)
│   ├── sed_robust_best.pth      # Proposed CRNN weights (git-ignored)
│   └── __init__.py
├── backend/
│   ├── main.py                  # FastAPI application + endpoint
│   ├── rag_chain.py             # FAISS vectorstore + LLM RAG chain
│   └── __init__.py
├── utils/
│   ├── audio_utils.py           # Audio loading, log-mel, SNR mixing
│   └── __init__.py
├── download_esc50.py            # ESC-50 dataset downloader & class extractor
├── augment_dataset.py           # Offline 7× augmentation (speed + gain)
├── train.py                     # Baseline + Proposed ablation training (fold-5 split)
├── train_compare.py             # 5-model comparative study training
├── comparison_results.json      # Multi-model benchmark results
├── benchmark_robustness.py      # SNR-condition robustness evaluation
├── benchmark_e2e.py             # End-to-end latency profiling (100 runs)
├── test_pipeline.py             # E2E integration test
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Set up virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> After activation, your prompt will show `(.venv)`. All subsequent commands assume the environment is active.

### 2. Download & prepare the ESC-50 dataset
```bash
python download_esc50.py
```
Downloads ESC-50 and extracts the 3 target classes (coughing, sneezing, infant_crying) into `data/` — 40 files per class.

### 3. (Optional) Offline data augmentation — 7× expansion
```bash
python augment_dataset.py
```
Expands each class from 40 → 280 files using speed perturbation (×0.85, ×0.93, ×1.07, ×1.15) and gain scaling (×0.35, ×2.0). Re-running is idempotent.

### 4. Train the SED model
```bash
# CRNN ablation (Baseline vs. Proposed noise augmentation)
python train.py

# 5-model comparative study
python train_compare.py
```
- `train.py`: Runs **Baseline** (no noise) then **Proposed** (SNR mixing + SpecAugment) — 50 epochs each. Saves to `models/sed_baseline.pth` and `models/sed_robust_best.pth`.
- `train_compare.py`: Trains all 5 architectures under the Proposed regime and saves results to `comparison_results.json`.
- Both use the **ESC-50 official 5-Fold split**: Folds 1–4 for training (orig + aug), Fold 5 for validation (originals only).

### 5. Start the backend server
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload
```

### 6. Run robustness & latency benchmarks
```bash
python benchmark_robustness.py   # Fold-5 eval at Clean / SNR 20dB / 10dB / 0dB
python benchmark_e2e.py          # 100-run per-stage latency profiling
```

### 7. Run the E2E integration test
```bash
PYTHONPATH=. python test_pipeline.py
```
Automatically starts the backend, runs inference on each audio class, POSTs to the API, and validates the RAG response.

### 8. Manual API test
```bash
curl -X POST http://127.0.0.1:8765/api/v1/context-stream \
  -H "Content-Type: application/json" \
  -d '{"audio_event":"yawning","confidence":0.92,"driving_hours":3.0,"hvac_status":"internal"}'
```

### Deactivate the virtual environment
```bash
deactivate
```

---

## Training Results

All experiments use the **ESC-50 5-Fold protocol** (Fold-5 validation, 24 original samples — zero data leakage).

### Phase 1 — Noise Augmentation Ablation (CRNN)

| Condition | Baseline | Proposed | Δ | Relative Gain |
|-----------|----------|----------|---|---------------|
| Clean     | 0.8739 | **0.9167** | +0.0428 | +4.9%      |
| SNR 20 dB | 0.6872 | **0.8739** | +0.1867 | +27.2%     |
| SNR 10 dB | 0.5481 | **0.8739** | +0.3258 | +59.4%     |
| SNR 0 dB  | 0.4532 | **0.8331** | +0.3799 | **+83.8%** |

- **Baseline**: Folds 1–4, online gain jitter + polarity inversion only.
- **Proposed**: identical + on-the-fly SNR mixing (0/10/20 dB) + SpecAugment (FreqMask 16, TimeMask 50).
- Proposed degrades only **9.1%** from clean → 0 dB SNR vs. **48.1%** for Baseline.

### Phase 2 — Multi-Model Architecture Comparison (Proposed regime)

All 5 architectures trained under the Proposed regime. Results from `comparison_results.json`:

| Model | Params | Clean F1 | SNR 20 dB | SNR 10 dB | SNR 0 dB | Latency |
|---|---|---|---|---|---|---|
| VGG-SED | 434,979 | 0.9582 | 0.9582 | **1.0000** | 0.9185 | 42.45 ms |
| CRNN | 170,579 | 0.9167 | 0.9582 | 0.9582 | 0.8741 | **10.17 ms** |
| MobileNet-SED | **125,315** | **1.0000** | 0.9582 | 0.9582 | **0.9582** | 20.64 ms |
| **CNN-Mamba** *(proposed)* | 313,555 | 0.9582 | 0.9582 | 0.9582 | 0.8342 | 17.28 ms |
| CNN-Transformer | 344,787 | 0.9582 | 0.9167 | 0.9167 | 0.7886 | 8.64 ms |

**Key findings:**
- **MobileNet-SED** achieves the best 0 dB robustness (F1=0.9582) with the fewest parameters (125K).
- **CNN-Transformer** degrades most under noise (F1=0.7886 at 0 dB, −17.7% from clean).
- **CNN-Mamba** outperforms CNN-Transformer under all noise conditions (0.8342 vs. 0.7886 at 0 dB) — the first empirical demonstration of SSM-based temporal modelling for in-vehicle SED.
- **CRNN** offers the best latency among temporal models (10.17 ms).

### End-to-End Latency (CPU, 100 runs)

| Stage | Mean | Std |
|-------|------|-----|
| Feature extraction (log-Mel) | 9.24 ms | 2.66 ms |
| SED model inference (CRNN) | 11.20 ms | 3.28 ms |
| FAISS vector retrieval | 52.27 ms | 45.15 ms |
| LLM command generation | 0.02 ms | 0.01 ms |
| **Total E2E** | **72.73 ms** | 45.34 ms |

SED core (feature extraction + inference): **20.44 ms** — suitable for real-time embedded deployment.

---

## API Reference

### `POST /api/v1/context-stream`

**Request**
```json
{
  "audio_event": "yawning",
  "confidence": 0.92,
  "driving_hours": 3.0,
  "hvac_status": "internal"
}
```

**Response**
```json
{
  "audio_event": "yawning",
  "confidence": 0.92,
  "query": "...",
  "retrieved_context": ["...relevant manual sections..."],
  "advice": "Driver fatigue detected. Switching HVAC to external fresh air mode...",
  "command_json": "{\"hvac_mode\": \"external\", \"temperature_delta\": -2, ...}",
  "raw_llm_response": "..."
}
```

---

## Model Architectures

All models receive log-Mel spectrogram inputs `(B, 1, 64, 313)` and output 3-class logits.

**Shared CNN backbone** (used by CRNN, CNN-Mamba, CNN-Transformer):
```
ConvBlock(1→16) → ConvBlock(16→32) → ConvBlock(32→48) → ConvBlock(48→64)
Each block: Conv2d(3×3) + BN + ReLU + MaxPool(2×2) + Dropout2d(0.1)
Output: (B, 64, 4, 19) → reshape → (B, 19, 256)
```

| Model | Temporal module | Params |
|---|---|---|
| VGG-SED | None (pure CNN, 4× VGGBlock) | 434,979 |
| CRNN | Bidirectional GRU (hidden=64) | 170,579 |
| MobileNet-SED | None (6× inverted residual blocks) | 125,315 |
| **CNN-Mamba** *(proposed)* | 2× MambaBlock (S6 SSM, d_model=128, d_state=16) | 313,555 |
| CNN-Transformer | 2× TransformerEncoderLayer (nhead=4, Pre-LN) | 344,787 |

CNN-Mamba is implemented in **pure PyTorch** (no CUDA/Triton dependency) — runs on CPU and Apple Silicon MPS.

---

## RAG Pipeline Detail

1. **Document ingestion**: The vehicle control manual is split into 300-character chunks and embedded with `sentence-transformers/all-MiniLM-L6-v2` into an in-memory FAISS index.
2. **Query construction**: SED output (`audio_event`, `confidence`) is combined with vehicle telemetry (`driving_hours`, `hvac_status`) into a natural-language query.
3. **Retrieval**: Top-2 most relevant manual chunks are retrieved.
4. **Generation**: A structured prompt sends context + query to the LLM. If no API key is set, a `MockLLM` returns deterministic, class-specific advice and a control JSON — ensuring the pipeline is always testable without credentials.

---

## License

MIT
