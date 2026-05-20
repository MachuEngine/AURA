"""Download ESC-50 dataset and organise into AURA data layout."""
import csv
import io
import os
import shutil
import zipfile

import requests

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZIP_FILE = "ESC-50-master.zip"
EXTRACT_DIR = "ESC-50-master"

# ESC-50 category → destination folder
TARGET_MAP = {
    "coughing":    "data/coughing",
    "sneezing":    "data/sneezing",
    "crying_baby": "data/infant_crying",
}

NOISE_MAP = {
    "engine":        "data/noise/vehicle_engine",
    "helicopter":    "data/noise/vehicle_engine",
    "airplane":      "data/noise/vehicle_engine",
    "rain":          "data/noise/environmental",
    "wind":          "data/noise/environmental",
    # ESC-50 has no 'white_noise'; vacuum_cleaner is the closest HVAC proxy
    "vacuum_cleaner": "data/noise/hvac",
}

ALL_MAP = {**TARGET_MAP, **NOISE_MAP}


def download_zip(url: str, dest: str):
    print(f"Downloading ESC-50 from GitHub...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  Download complete: {downloaded / 1e6:.1f} MB      ")


def main():
    # Clear existing data dirs that will be repopulated
    for dest in ALL_MAP.values():
        if os.path.exists(dest):
            shutil.rmtree(dest)
    for dest in ALL_MAP.values():
        os.makedirs(dest, exist_ok=True)

    # Download zip
    if not os.path.exists(ZIP_FILE):
        download_zip(ESC50_ZIP_URL, ZIP_FILE)
    else:
        print(f"Found existing {ZIP_FILE}, skipping download.")

    # Extract
    print("Extracting archive...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extractall(".")
    print("Extraction complete.")

    # Read metadata CSV
    csv_path = os.path.join(EXTRACT_DIR, "meta", "esc50.csv")
    audio_src = os.path.join(EXTRACT_DIR, "audio")

    counts = {cat: 0 for cat in ALL_MAP}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            if cat not in ALL_MAP:
                continue
            src = os.path.join(audio_src, row["filename"])
            dst = os.path.join(ALL_MAP[cat], row["filename"])
            shutil.copy2(src, dst)
            counts[cat] += 1

    # Report
    print("\nFiles organised:")
    for cat, dest in ALL_MAP.items():
        print(f"  {cat:20s} → {dest:35s} ({counts[cat]} files)")

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(EXTRACT_DIR)
    os.remove(ZIP_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
