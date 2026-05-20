"""End-to-End pipeline test: SED inference → RAG backend POST → result validation."""
import os
import sys
import time
import subprocess
import signal
import json
import glob
import torch
import httpx
import numpy as np

from models.sed_model import CRNN
from utils.audio_utils import load_audio, waveform_to_logmel, build_mel_transform, N_MELS

BACKEND_URL = "http://127.0.0.1:8765"
WEIGHTS_PATH = "models/sed_model_best.pth"
FIXED_FRAMES = 128

CLASS_LABELS = {0: "background_noise", 1: "coughing", 2: "yawning", 3: "infant_crying"}
AUDIO_CLASSES_FOR_BACKEND = {
    0: "background_noise",
    1: "coughing",
    2: "yawning",
    3: "infant_crying",
}


def load_model():
    model = CRNN(num_classes=4, n_mels=N_MELS)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    print(f"[STEP 1] Model loaded from {WEIGHTS_PATH}")
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

    x = logmel.unsqueeze(0)  # (1, 1, n_mels, T)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        class_id = probs.argmax(dim=1).item()
        confidence = probs[0, class_id].item()

    class_name = CLASS_LABELS[class_id]
    print(f"[STEP 1] Inference: {wav_path} → {class_name} (conf={confidence:.3f})")
    return class_name, confidence


def start_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
         "--port", "8765", "--log-level", "warning"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[STEP 2] Backend server starting...")
    return proc


def wait_for_server(timeout=30):
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
                    driving_hours: float = 2.5, hvac_status: str = "internal"):
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
    all_passed = True

    # Step 1: Load model
    model = load_model()

    # Step 2: Start server
    server_proc = start_server()
    try:
        assert wait_for_server(), "Server failed to start within timeout"

        # Step 3: Pick one wav per class and run full E2E
        test_cases = [
            ("data/coughing/coughing_000.wav", "coughing"),
            ("data/yawning/yawning_000.wav", "yawning"),
            ("data/infant_crying/infant_crying_000.wav", "infant_crying"),
            ("data/background_noise/background_noise_000.wav", "background_noise"),
        ]

        for wav_path, expected_class in test_cases:
            print(f"\n--- Test: {wav_path} ---")
            # SED inference
            class_name, confidence = infer_audio(model, wav_path)

            # POST to backend (use inferred class for realistic E2E)
            result = post_to_backend(class_name, confidence)
            print(f"[STEP 3] RAG response for '{class_name}':")
            print(f"  Advice   : {result['advice'][:120]}...")
            print(f"  Command  : {result['command_json'][:80]}")
            assert result["audio_event"] == class_name
            assert result["confidence"] == confidence
            assert len(result["advice"]) > 0
            assert result["command_json"].strip() != ""
            print(f"  [PASS]")

        print("\n=============================")
        print("ALL E2E TESTS PASSED")
        print("=============================")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        all_passed = False
    finally:
        server_proc.terminate()
        server_proc.wait()
        print("[CLEANUP] Backend server stopped.")

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
