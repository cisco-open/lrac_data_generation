# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import subprocess
import requests
import hashlib
from pathlib import Path

# --- Configuration ---
# Directories
output_dir = Path("./fma")
tmp_dir = Path("tmp")
resampled_dir = output_dir / "resampled" / "noise"

# FMA Download URLs and File Info
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_MEDIUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
FMA_MEDIUM_SHA1 = "c67b69ea232021025fca9231fc1c7c1a063ab50b"

curation_file = Path("./datafiles/fma/train_meta_curated.csv")

# --- Helper Functions ---

def run_external_command(command: list, env_vars: dict = None):
    """Helper function to run external commands."""
    print(f"Running command: {' '.join(map(str, command))}")
    process_env = os.environ.copy()
    if env_vars:
        process_env.update(env_vars)
    try:
        result = subprocess.run(
            command,
            check=True,
            env=process_env,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(map(str, command))}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        exit(1)

# --- Main Script Logic ---

if __name__ == "__main__":
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resampled_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparing FMA data (DNS5-style workflow) ===")

    # 1. Download and Extract Data
    print("[FMA] Downloading and extracting data")
    download_done_file = output_dir / "download_fma.done"

    if download_done_file.exists():
        print("Skip downloading FMA as it has already finished.")
    else:
        # Download files
        for url in [FMA_METADATA_URL, FMA_MEDIUM_URL]:
            local_path = output_dir / Path(url).name
            if not local_path.exists():
                print(f"Downloading {url} to {local_path}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Finished downloading {local_path.name}")
            else:
                print(f"File already exists, skipping download: {local_path}")

        # Verify checksum for fma_medium.zip
        medium_zip_path = output_dir / "fma_medium.zip"
        print(f"Verifying SHA1 checksum for {medium_zip_path.name}...")
        sha1 = hashlib.sha1()
        with open(medium_zip_path, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        
        if sha1.hexdigest() == FMA_MEDIUM_SHA1:
            print("Checksum verification successful.")
        else:
            print(f"Error: Checksum mismatch for {medium_zip_path.name}.")
            exit(1)

        # Extract archives using 7z
        for zip_name in ["fma_metadata.zip", "fma_medium.zip"]:
            zip_path = output_dir / zip_name
            print(f"Extracting {zip_path}...")
            # Check if extraction target exists to avoid re-extraction
            if zip_name == "fma_medium.zip" and (output_dir / "fma_medium").exists():
                 print(f"Extraction directory for {zip_name} already exists. Skipping.")
            else:
                run_external_command(["7z", "x", zip_path, f"-o{output_dir}"])
                print(f"Removing {zip_path}")
                zip_path.unlink()

        download_done_file.touch()

    # 2. Generate SCP file for all raw FMA audio files
    raw_scp_file = tmp_dir / "fma_noise.scp"
    if raw_scp_file.exists():
        print(f"Raw FMA scp file already exists. Skipping generation: {raw_scp_file}")
    else:
        print("[FMA] Generating scp file for all raw audio")
        audio_base_dir = output_dir / "fma_medium"
        audio_files = sorted(list(audio_base_dir.rglob("*.mp3")))
        print(f"Found {len(audio_files)} raw FMA .mp3 files.")
        
        with open(raw_scp_file, "w") as f_scp:
            for audio_path in audio_files:
                # FMA filenames are numeric track IDs, e.g., '000002.mp3'
                file_id = f"fma_{audio_path.stem}"
                f_scp.write(f"{file_id} {audio_path.resolve()}\n")
        print(f"Created raw FMA scp file: {raw_scp_file}")

    # 3. Filter using the curation list
    filtered_scp_file = tmp_dir / "fma_noise_filtered_curation.scp"
    if filtered_scp_file.exists():
        print(f"Filtered scp file already exists. Delete {filtered_scp_file} to re-filter.")
    else:
        if not curation_file.exists():
            print(f"Error: Curation file not found at {curation_file}")
            print("Please create this file with the FMA track IDs you want to keep.")
            exit(1)
            
        print("[FMA] Filtering using curation list")
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
    resampled_scp_file = tmp_dir / f"fma_noise_resampled_filtered_curation.scp"
    if resampled_scp_file.exists():
        print(f"Resampled scp file already exists. Delete {resampled_scp_file} to re-resample.")
    else:
        print(f"[FMA] Resampling to 24kHz sampling rate")
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

    print("\n--- FMA data preparation finished ---")
    print(f"Final output file: {resampled_scp_file}")