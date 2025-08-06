# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import subprocess
import requests
from pathlib import Path

# --- Configuration ---
# Directories
output_dir = Path("./wham_noise_48k")
tmp_dir = Path("tmp")
resampled_dir = output_dir / "resampled" / "audio"

# WHAM! Download URL
WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip"

curation_file = Path("./datafiles/wham/train_meta_curated.csv")

# --- Helper Functions ---

def run_external_command(command: list, working_dir: Path = None, env_vars: dict = None):
    """Helper function to run external commands."""
    print(f"Running command: {' '.join(map(str, command))}")
    process_env = os.environ.copy()
    if env_vars:
        process_env.update(env_vars)
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=working_dir,
            env=process_env,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command in '{working_dir or '.'}': {' '.join(map(str, command))}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        exit(1)

def download_file(url: str, local_path: Path):
    """Downloads a single file."""
    if local_path.exists():
        print(f"File already exists, skipping download: {local_path}")
        return
    print(f"Downloading {url} to {local_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Finished downloading {local_path.name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        exit(1)

# --- Main Script Logic ---

if __name__ == "__main__":
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resampled_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing WHAM! noise data (DNS5-style workflow) ===")

    # 1. Download and Extract Data
    print("[WHAM!] Downloading and extracting data")
    download_done_file = output_dir / "download_high_res_wham.done"

    if download_done_file.exists():
        print("Skip downloading WHAM! as it has already finished.")
    else:
        wham_zip_path = output_dir / "high_res_wham.zip"
        download_file(WHAM_URL, wham_zip_path)
        
        print(f"Extracting {wham_zip_path}...")
        run_external_command(["unzip", "-o", wham_zip_path.name], working_dir=output_dir)

        download_done_file.touch()

    # 2. Generate SCP file for all raw WHAM! audio files
    raw_scp_file = tmp_dir / "wham_noise.scp"
    if raw_scp_file.exists():
        print(f"Raw WHAM! scp file already exists. Skipping generation: {raw_scp_file}")
    else:
        print("[WHAM!] Generating scp file for all raw audio")
        audio_base_dir = output_dir / "high_res_wham" / "audio"
        if not audio_base_dir.exists():
            print(f"Error: Audio directory not found at {audio_base_dir}")
            exit(1)
            
        audio_files = sorted(list(audio_base_dir.rglob("*.wav")))
        print(f"Found {len(audio_files)} raw WHAM! .wav files.")
        
        with open(raw_scp_file, "w") as f_scp:
            for audio_path in audio_files:
                file_id = f"{audio_path.stem}"
                f_scp.write(f"{file_id} {audio_path.resolve()}\n")
        print(f"Created raw WHAM! scp file: {raw_scp_file}")

    # 3. Filter using the curation list
    filtered_scp_file = tmp_dir / "wham_noise_filtered_curation.scp"
    if filtered_scp_file.exists():
        print(f"Filtered scp file already exists. Delete {filtered_scp_file} to re-filter.")
    else:
        if not curation_file.exists():
            print(f"Error: Curation file not found at {curation_file}")
            print("Please create this file with the WHAM! file IDs you want to keep.")
            exit(1)
            
        print("[WHAM!] Filtering using curation list")
        run_external_command(
            [
                "python", "utils/filter_via_curation_list.py",
                "--scp_path", str(raw_scp_file),
                "--curation_path", str(curation_file),
                "--outfile", str(filtered_scp_file),
            ],
            env_vars={"OMP_NUM_THREADS": "1"}
        )

    # 4. Resample the filtered files to a single sampling rate
    resampled_scp_file = tmp_dir / f"wham_noise_resampled_filtered_curation.scp"
    if resampled_scp_file.exists():
        print(f"Resampled scp file already exists. Delete {resampled_scp_file} to re-resample.")
    else:
        print(f"[WHAM!] Resampling to 24kHz sampling rate")
        run_external_command(
            [
                "python", "utils/resample_to_single_fs.py",
                "--in_scpfile", str(filtered_scp_file),
                "--out_fs", "24000",
                "--out_scpfile", str(resampled_scp_file),
                "--outdir", str(resampled_dir),
                "--max_files", "5000",
                "--nj", "8",
                "--chunksize", "1000"
            ],
            env_vars={"OMP_NUM_THREADS": "1"}
        )

    print("\n--- WHAM! data preparation finished ---")
    print(f"Final output file: {resampled_scp_file}")