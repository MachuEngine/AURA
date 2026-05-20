# AURA — Audio Understanding & RAG for Auto

An in-vehicle AI framework that continuously listens for acoustic events, classifies them with a **sub-1M-parameter CRNN**, and generates context-aware vehicle control commands through a **Retrieval-Augmented Generation (RAG)** pipeline.

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
| **Model size** | 170,708 parameters — well under the 1M embedded target |
| **Inference latency** | ~2ms on CPU (MacBook M-series) |
| **Sound classes** | coughing, sneezing, infant crying (3-class, ESC-50 subset) |
| **Data augmentation** | Offline 7× (speed ×4 + gain ×2) + Online SpecAugment (FreqMask + TimeMask) |
| **RAG retrieval** | FAISS + `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM fallback** | Automatic `MockLLM` when no API key is present |
| **API** | RESTful FastAPI with Pydantic v2 validation |

---

## Project Structure

```
AURA/
├── data/                        # Dummy audio dataset (git-ignored, regenerate below)
│   ├── background_noise/
│   ├── coughing/
│   ├── yawning/
│   └── infant_crying/
├── models/
│   ├── sed_model.py             # CRNN architecture definition
│   └── __init__.py
├── backend/
│   ├── main.py                  # FastAPI application + endpoint
│   ├── rag_chain.py             # FAISS vectorstore + LLM RAG chain
│   └── __init__.py
├── utils/
│   ├── audio_utils.py           # Audio loading & log-mel transform
│   └── __init__.py
├── download_esc50.py            # ESC-50 dataset downloader & class extractor
├── augment_dataset.py           # Offline 7× augmentation (speed + gain)
├── train.py                     # CRNN training script (+ online SpecAugment)
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
python train.py
```
- Runs for 5 epochs; logs F1-score and inference latency per epoch to `training.log`.
- Saves best weights to `models/sed_model_best.pth`.
- Online SpecAugment is applied during training: `FrequencyMasking(12)` + `TimeMasking(40)`.

### 5. Start the backend server
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload
```

### 6. Run the E2E test
```bash
PYTHONPATH=. python test_pipeline.py
```
Automatically starts the backend, runs inference on each audio class, POSTs to the API, and validates the RAG response.

### 7. Manual API test
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

Augmentation effect on Val F1 (10 epochs, ESC-50 3-class):

| Setup | Samples | Best Val F1 | Final Loss |
|-------|---------|-------------|------------|
| Originals only | 120 | 0.7980 | 0.9082 |
| + Offline 7× augmentation | 840 | **0.8904** | **0.2463** |

Offline augmentation (`augment_dataset.py`) + online SpecAugment combined yielded an **+11% absolute F1 gain**. The model also converged steadily across all 10 epochs with the larger dataset, versus plateauing at epoch 4 on originals only.

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

## Model Architecture

```
Input: (B, 1, 64, 128)  — batch × channel × mel_bins × time_frames

CNN Backbone:
  ConvBlock(1  → 16):  Conv2d + BN + ReLU + MaxPool(2×2) + Dropout2d
  ConvBlock(16 → 32):  Conv2d + BN + ReLU + MaxPool(2×2) + Dropout2d
  ConvBlock(32 → 48):  Conv2d + BN + ReLU + MaxPool(2×2) + Dropout2d
  ConvBlock(48 → 64):  Conv2d + BN + ReLU + MaxPool(2×2) + Dropout2d

Output: (B, 64, 4, T')  →  reshape to (B, T', 256)

Bi-GRU:  hidden=64, bidirectional → output (B, T', 128)
Global average pooling → (B, 128)

Classifier: Dropout(0.3) + Linear(128 → 4)

Total parameters: 170,708  (<1M ✓)
```

---

## RAG Pipeline Detail

1. **Document ingestion**: The vehicle control manual is split into 300-character chunks and embedded with `sentence-transformers/all-MiniLM-L6-v2` into an in-memory FAISS index.
2. **Query construction**: SED output (`audio_event`, `confidence`) is combined with vehicle telemetry (`driving_hours`, `hvac_status`) into a natural-language query.
3. **Retrieval**: Top-2 most relevant manual chunks are retrieved.
4. **Generation**: A structured prompt sends context + query to the LLM. If no API key is set, a `MockLLM` returns deterministic, class-specific advice and a control JSON — ensuring the pipeline is always testable without credentials.

---

## License

MIT
