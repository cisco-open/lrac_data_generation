# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import subprocess
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# Directories
output_dir = Path("./fsd50k")
tmp_dir = Path("tmp")
resampled_dir = output_dir / "resampled" / "FSD50K.dev_audio"

# Zenodo base URL for FSD50K files
BASE_URL = "https://zenodo.org/records/4060432/files"

# Files to download
# Note: Audio files are handled separately due to being split
METADATA_FILES = {
    "FSD50K.ground_truth.zip": f"{BASE_URL}/FSD50K.ground_truth.zip?download=1",
    "FSD50K.metadata.zip": f"{BASE_URL}/FSD50K.metadata.zip?download=1",
    "FSD50K.doc.zip": f"{BASE_URL}/FSD50K.doc.zip?download=1",
}

curation_file = Path("./datafiles/fsd50k/train_meta_curated.csv")

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

    print("=== Preparing FSD50K data (DNS5-style workflow) ===")

    # 1. Download and Extract Data
    print("[FSD50K] Downloading and extracting data")
    download_done_file = output_dir / "download_fsd50k.done"

    if download_done_file.exists():
        print("Skip downloading FSD50K as it has already finished.")
    else:
        # Download metadata files
        for filename, url in METADATA_FILES.items():
            zip_path = output_dir / filename
            download_file(url, zip_path)
            print(f"Extracting {zip_path}...")
            run_external_command(["unzip", "-o", zip_path.name], working_dir=output_dir)

        # Download split audio files
        audio_files_to_download = [f"{BASE_URL}/FSD50K.dev_audio.zip?download=1"]
        audio_files_to_download.extend([f"{BASE_URL}/FSD50K.dev_audio.z0{i}?download=1" for i in range(1, 6)])
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(download_file, url, output_dir / Path(url).name.split('?')[0]) for url in audio_files_to_download]
            for future in futures:
                future.result() # Wait for all downloads to complete

        # Reassemble and extract the main audio archive
        print("Reassembling and extracting FSD50K audio archive...")
        unsplit_zip = "unsplit.zip"
        run_external_command(
            ["zip", "-s", "0", "FSD50K.dev_audio.zip", "--out", unsplit_zip],
            working_dir=output_dir
        )
        run_external_command(["unzip", "-o", unsplit_zip], working_dir=output_dir)

        # Clone Audioset ontology
        ontology_dir = output_dir / "ontology"
        if not ontology_dir.exists():
            print("Cloning Audioset ontology...")
            run_external_command(["git", "clone", "https://github.com/audioset/ontology.git", str(ontology_dir)])
        else:
            print("Ontology directory already exists.")

        # Clean up downloaded archives
        print("Cleaning up downloaded archives...")
        for f in output_dir.glob("FSD50K.*.zip"): f.unlink()
        for f in output_dir.glob("FSD50K.*.z*"): f.unlink()
        if (output_dir / unsplit_zip).exists(): (output_dir / unsplit_zip).unlink()

        download_done_file.touch()

    # 2. Generate SCP file for all raw FSD50K audio files
    raw_scp_file = tmp_dir / "fsd50k_noise.scp"
    if raw_scp_file.exists():
        print(f"Raw FSD50K scp file already exists. Skipping generation: {raw_scp_file}")
    else:
        print("[FSD50K] Generating scp file for all raw audio")
        audio_base_dir = output_dir / "FSD50K.dev_audio"
        audio_files = sorted(list(audio_base_dir.rglob("*.wav")))
        print(f"Found {len(audio_files)} raw FSD50K .wav files.")
        
        with open(raw_scp_file, "w") as f_scp:
            for audio_path in audio_files:
                file_id = f"fsd50k_{audio_path.stem}"
                f_scp.write(f"{file_id} {audio_path.resolve()}\n")
        print(f"Created raw FSD50K scp file: {raw_scp_file}")

    # 3. Filter using the curation list
    filtered_scp_file = tmp_dir / "fsd50k_noise_filtered_curation.scp"
    if filtered_scp_file.exists():
        print(f"Filtered scp file already exists. Delete {filtered_scp_file} to re-filter.")
    else:
        if not curation_file.exists():
            print(f"Error: Curation file not found at {curation_file}")
            print("Please create this file with the FSD50K file IDs you want to keep.")
            exit(1)
            
        print("[FSD50K] Filtering using curation list")
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
    resampled_scp_file = tmp_dir / f"fsd50k_noise_resampled_filtered_curation.scp"
    if resampled_scp_file.exists():
        print(f"Resampled scp file already exists. Delete {resampled_scp_file} to re-resample.")
    else:
        print(f"[FSD50K] Resampling to 24kHz sampling rate")
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

    print("\n--- FSD50K data preparation finished ---")
    print(f"Final output file: {resampled_scp_file}")