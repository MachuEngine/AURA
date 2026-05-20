"""End-to-End pipeline test: SED inference → RAG backend POST → result validation."""
import glob
import os
import subprocess
import sys
import time

import httpx
import torch

from models.sed_model import CRNN
from utils.audio_utils import (
    FIXED_FRAMES,
    N_MELS,
    build_mel_transform,
    load_audio,
    waveform_to_logmel,
)

BACKEND_URL = "http://127.0.0.1:8765"
WEIGHTS_PATH = "models/sed_model_best.pth"

CLASS_LABELS = {0: "coughing", 1: "sneezing", 2: "infant_crying"}


def first_wav(directory: str) -> str:
    files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    assert files, f"No WAV files in {directory}"
    return files[0]


def load_model():
    model = CRNN(num_classes=3, n_mels=N_MELS)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    print(f"[STEP 1] Model loaded  ({WEIGHTS_PATH})")
    return model


def infer_audio(model, wav_path: str):
    mel_transform = build_mel_transform()
    waveform = load_audio(wav_path)
    logmel = waveform_to_logmel(waveform, mel_transform)

    T = logmel.shape[-1]
    if T >= FIXED_FRAMES:
        logmel = logmel[:, :, :FIXED_FRAMES]
    else:
        logmel = torch.nn.functional.pad(logmel, (0, FIXED_FRAMES - T))

    with torch.no_grad():
        logits = model(logmel.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        class_id = probs.argmax(dim=1).item()
        confidence = round(probs[0, class_id].item(), 4)

    class_name = CLASS_LABELS[class_id]
    print(f"[STEP 1] Inference: {os.path.basename(wav_path)} → {class_name} (conf={confidence:.3f})")
    return class_name, confidence


def start_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", "8765", "--log-level", "warning"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[STEP 2] Backend server starting...")
    return proc


def wait_for_server(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BACKEND_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[STEP 2] Server ready: {r.json()}")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def post_to_backend(audio_event: str, confidence: float,
                    driving_hours: float = 2.5, hvac_status: str = "internal") -> dict:
    payload = {
        "audio_event": audio_event,
        "confidence": confidence,
        "driving_hours": driving_hours,
        "hvac_status": hvac_status,
    }
    r = httpx.post(f"{BACKEND_URL}/api/v1/context-stream", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def run_tests():
    model = load_model()
    server_proc = start_server()

    try:
        assert wait_for_server(), "Server failed to start within timeout"

        test_cases = [
            (first_wav("data/coughing"),      "coughing"),
            (first_wav("data/sneezing"),       "sneezing"),
            (first_wav("data/infant_crying"),  "infant_crying"),
        ]

        for wav_path, expected_class in test_cases:
            print(f"\n--- Test: {expected_class} ---")
            class_name, confidence = infer_audio(model, wav_path)

            result = post_to_backend(class_name, confidence)
            print(f"[STEP 3] RAG response for '{class_name}':")
            print(f"  Advice : {result['advice'][:120]}...")
            print(f"  Command: {result['command_json'][:80]}")

            assert result["audio_event"] == class_name
            assert result["confidence"] == confidence
            assert len(result["advice"]) > 0
            assert result["command_json"].strip() not in ("", "{}")
            print(f"  [PASS]")

        print("\n=============================")
        print("ALL E2E TESTS PASSED")
        print("=============================")
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False
    finally:
        server_proc.terminate()
        server_proc.wait()
        print("[CLEANUP] Backend server stopped.")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
