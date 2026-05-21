"""End-to-end latency benchmark: 100-run profiling of each pipeline stage.

Stages measured:
  1. Feature extraction  (load_audio + waveform_to_logmel)
  2. SED model inference (CRNN forward pass)
  3. FAISS vector search (retriever.invoke)
  4. LLM response gen   (MockLLM.invoke)

Results are saved to latency_results.json.
"""
import json
import statistics
import time

import torch

from backend.rag_chain import build_rag_chain
from models.sed_model import CRNN
from utils.audio_utils import (
    FIXED_FRAMES,
    N_MELS,
    build_mel_transform,
    load_audio,
    waveform_to_logmel,
)
import glob
import os

N_RUNS = 100
WEIGHTS_PATH = "models/sed_robust_best.pth"
CLASS_LABELS = {0: "coughing", 1: "sneezing", 2: "infant_crying"}


def first_wav(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    assert files, f"No WAV files in {directory}"
    return files[0]


def pad_or_truncate(logmel):
    t = logmel.shape[-1]
    if t >= FIXED_FRAMES:
        return logmel[:, :, :FIXED_FRAMES]
    return torch.nn.functional.pad(logmel, (0, FIXED_FRAMES - t))


def measure_stage(fn, n_runs, warmup=10):
    """Return (mean_ms, std_ms) over n_runs after warmup calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times)


def main():
    device = torch.device("cpu")  # CPU — matches embedded target

    # --- Load model ---
    model = CRNN(num_classes=3, n_mels=N_MELS).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Model loaded: {WEIGHTS_PATH}")

    # --- Load RAG chain ---
    print("Initialising RAG chain (FAISS + MockLLM)...")
    retriever, llm = build_rag_chain()
    print("RAG chain ready.")

    # --- Prepare audio ---
    wav_path = first_wav("data/coughing")
    mel_transform = build_mel_transform()

    # Pre-compute logmel so stage 1 and 2 can be measured independently
    waveform_cache = load_audio(wav_path)
    logmel_cache = pad_or_truncate(waveform_to_logmel(waveform_cache, mel_transform))
    input_tensor = logmel_cache.unsqueeze(0).to(device)

    # Stage 1: Feature extraction
    def stage_feat():
        wf = load_audio(wav_path)
        lm = waveform_to_logmel(wf, mel_transform)
        pad_or_truncate(lm)

    mean1, std1 = measure_stage(stage_feat, N_RUNS)
    print(f"Stage 1 (feature extraction):  {mean1:.2f} ± {std1:.2f} ms")

    # Stage 2: SED model inference
    def stage_infer():
        with torch.no_grad():
            logits = model(input_tensor)
            _ = logits.argmax(dim=1).item()

    mean2, std2 = measure_stage(stage_infer, N_RUNS)
    print(f"Stage 2 (SED inference):        {mean2:.2f} ± {std2:.2f} ms")

    # Stage 3: FAISS vector search
    query = "Sound event: coughing (confidence: 0.97). Driver has been driving for 2.5 hours. Current HVAC mode: internal."

    def stage_faiss():
        retriever.invoke(query)

    mean3, std3 = measure_stage(stage_faiss, N_RUNS)
    print(f"Stage 3 (FAISS retrieval):      {mean3:.2f} ± {std3:.2f} ms")

    # Stage 4: LLM response generation (MockLLM or real)
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"You are an intelligent in-vehicle AI assistant.\n\n"
        f"Relevant vehicle manual sections:\n{context}\n\n"
        f"Situation: {query}\n\n"
        f"Provide:\n1. ADVICE: ...\n2. COMMAND: ...\n"
    )

    def stage_llm():
        resp = llm.invoke(prompt)
        _ = resp.content if hasattr(resp, "content") else str(resp)

    mean4, std4 = measure_stage(stage_llm, N_RUNS)
    print(f"Stage 4 (LLM generation):       {mean4:.2f} ± {std4:.2f} ms")

    total_mean = mean1 + mean2 + mean3 + mean4
    total_std = (std1**2 + std2**2 + std3**2 + std4**2) ** 0.5

    print(f"\n{'='*50}")
    print(f"Total E2E latency:              {total_mean:.2f} ± {total_std:.2f} ms")
    print(f"{'='*50}")

    results = {
        "n_runs": N_RUNS,
        "device": str(device),
        "stages": {
            "feature_extraction_ms": {"mean": round(mean1, 3), "std": round(std1, 3)},
            "sed_inference_ms":       {"mean": round(mean2, 3), "std": round(std2, 3)},
            "faiss_retrieval_ms":     {"mean": round(mean3, 3), "std": round(std3, 3)},
            "llm_generation_ms":      {"mean": round(mean4, 3), "std": round(std4, 3)},
        },
        "total_e2e_ms": {"mean": round(total_mean, 3), "std": round(total_std, 3)},
    }
    with open("latency_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("Latency results saved to latency_results.json")


if __name__ == "__main__":
    main()
