# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import subprocess
import tarfile
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import glob

output_dir = Path("./dns5_fullband")
base_download_url = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
tmp_dir = Path("tmp")
curation_file = './datafiles/dns5_noise/train_meta_curated.csv'

print(f"Creating output directory: {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

print("=== Preparing DNS5 noise and RIR data ===")
print("[DNS5 noise and RIR] downloading")

blob_names_to_download = [
    "noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
    "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
    "datasets_fullband.impulse_responses_000.tar.bz2",
]

def download_file(blob_name: str):
    """Downloads a single file from the blob storage."""
    url = f"{base_download_url}/{blob_name}"
    local_path = output_dir / blob_name

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"File already exists, skipping: {local_path}")
        return

    try:
        print(f"Downloading {url} to {local_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Finished downloading {blob_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {blob_name}: {e}")
        raise

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(download_file, blob_name) for blob_name in blob_names_to_download]
    for future in futures:
        try:
            future.result()
        except Exception as exc:
            print(f"Download generated an exception: {exc}")
            exit(1)

print("Extracting archives...")
transform_prefix = "datasets_fullband/noise_fullband/"

for sub in ["audioset", "freesound"]:
    n = 6 if sub == "audioset" else 1
    for idx in range(n + 1):
        archive_name = f"noise_fullband/datasets_fullband.noise_fullband.{sub}_{idx:03d}.tar.bz2"
        archive_path = output_dir / archive_name
        xdir = output_dir / "datasets_fullband" / "noise_fullband" / f"{sub}_{idx:03d}"
        
        if xdir.exists() and any(xdir.iterdir()):
            print(f"Directory already exists and is not empty, skipping extraction: {xdir}")
            continue
            
        xdir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {archive_path}")
        try:
            with tarfile.open(archive_path, "r:bz2") as tar:
                for member in tar.getmembers():
                    if member.isreg():
                        relative_path_in_archive = Path(member.name)
                        if str(relative_path_in_archive).startswith(transform_prefix):
                            transformed_name = relative_path_in_archive.relative_to(transform_prefix)
                        else:
                            transformed_name = relative_path_in_archive

                        target_file_path = xdir / transformed_name
                        target_file_path.parent.mkdir(parents=True, exist_ok=True)

                        with tar.extractfile(member) as source_file:
                            if source_file:
                                with open(target_file_path, "wb") as dest_file:
                                    dest_file.write(source_file.read())
        except tarfile.ReadError as e:
            print(f"Error extracting {archive_path}: {e}")
        except FileNotFoundError:
            print(f"Archive not found, skipping extraction: {archive_path}")
        except Exception as e:
            print(f"An unexpected error occurred during extraction of {archive_path}: {e}")
            exit(1)

rir_archive_name = "datasets_fullband.impulse_responses_000.tar.bz2"
rir_archive_path = output_dir / rir_archive_name
rir_extract_dir = output_dir

print(f"Extracting {rir_archive_path}")
rir_extracted_check_dir = rir_extract_dir / "datasets_fullband" / "impulse_responses"

if rir_extracted_check_dir.exists() and any(rir_extracted_check_dir.iterdir()):
    print(f"RIR directory already exists and is not empty, skipping extraction: {rir_extracted_check_dir}")
else:
    try:
        with tarfile.open(rir_archive_path, "r:bz2") as tar:
            tar.extractall(path=rir_extract_dir)
    except tarfile.ReadError as e:
        print(f"Error extracting {rir_archive_path}: {e}")
    except FileNotFoundError:
        print(f"Archive not found, skipping extraction: {rir_archive_path}")
    except Exception as e:
        print(f"An unexpected error occurred during extraction of {rir_archive_path}: {e}")
        exit(1)

noise_scp_file = tmp_dir / "dns5_noise.scp"
if noise_scp_file.exists():
    print(f"Noise scp file already exists. Skipping generation: {noise_scp_file}")
else:
    print("[DNS5 noise and RIR] Generating scp file for noise")
    noise_base_dir = output_dir / "datasets_fullband" / "noise_fullband"
    noise_lines = []
    if noise_base_dir.exists():
        wav_files = sorted(list(noise_base_dir.rglob("*.wav")))
        print(f"Found {len(wav_files)} raw noise .wav files.")
        for wav_path in wav_files:
            try:
                file_id = wav_path.stem
                noise_lines.append(f"{file_id} {wav_path.resolve()}\n")
            except ValueError:
                print(f"Warning: Could not determine relative path for {wav_path}. Skipping.")
                continue

        with open(noise_scp_file, "w") as f_scp:
            f_scp.writelines(noise_lines)
        print(f"Created noise scp file: {noise_scp_file}")
    else:
        print(f"Error: Noise base directory not found: {noise_base_dir}. Cannot generate raw noise scp.")
        exit(1)

filtered_scp_file = tmp_dir / "dns5_noise_filtered_curation.scp"
resamp_scp_file = tmp_dir / "dns5_noise_resampled_filtered_curation.scp"

def run_external_python_script(script_path: str, args: list, env_vars: dict = None):
    """Helper function to run external Python scripts."""
    full_command = ["python", script_path] + args
    print(f"Running command: {' '.join(full_command)}")
    process_env = os.environ.copy()
    if env_vars:
        process_env.update(env_vars)
    try:
        subprocess.run(full_command, check=True, env=process_env, capture_output=True, text=True)
        print(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        exit(1)

if filtered_scp_file.exists():
    print(f"Filtered scp file already exists. Delete {filtered_scp_file} if you want to re-estimate.")
else:
    print("[DNS5 noise and RIR] filtering using curation lists")
    run_external_python_script(
        "utils/filter_via_curation_list.py",
        [
            "--scp_path", str(noise_scp_file),
            "--curation_path", str(curation_file),
            "--outfile", str(filtered_scp_file)
        ],
        env_vars={"OMP_NUM_THREADS": "1"}
    )

if resamp_scp_file.exists():
    print(f"Resampled scp file already exists. Delete {resamp_scp_file} if you want to re-resample.")
else:
    print("[DNS5 noise and RIR] resampling to 24kHz sampling rate")
    run_external_python_script(
        "utils/resample_to_single_fs.py",
        [
            "--in_scpfile", str(filtered_scp_file),
            "--out_fs", "24000",
            "--out_scpfile", str(resamp_scp_file),
            "--outdir", str(output_dir / "resampled" / "noise"),
            "--max_files", "5000",
            "--nj", "8",
            "--chunksize", "1000"
        ],
        env_vars={"OMP_NUM_THREADS": "1"}
    )

print("[DNS5 noise and RIR] preparing data files")

noise_data_by_fs = defaultdict(list)
try:
    with open(resamp_scp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                sampling_rate = parts[1]
                noise_data_by_fs[sampling_rate].append(line)
except FileNotFoundError:
    print(f"Error: Required file {resamp_scp_file} not found. Cannot split noise data.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading {resamp_scp_file}: {e}")
    exit(1)

all_files = []
for fs, lst in noise_data_by_fs.items():
    all_files.extend(lst)

with open(tmp_dir / "dns5_noise_resampled_filtered_curation.scp", "w") as f:
    for line in sorted(all_files):
        f.write(line)
print("Created dns5_noise_resampled_filtered_curation.scp")

print("--- Starting RIR Data Processing ---")
rir_scp_file = tmp_dir / "dns5_rirs_raw.scp"
if not rir_scp_file.exists():
    print("[RIR] Generating raw scp file...")
    rir_base_dir = output_dir / "datasets_fullband" / "impulse_responses"
    if rir_base_dir.exists():
        with open(rir_scp_file, "w") as f_scp:
            for wav_path in sorted(rir_base_dir.rglob("*.wav")):
                file_id = wav_path.stem
                f_scp.write(f"{file_id} {wav_path.resolve()}\n")
        print(f"Created raw RIR scp: {rir_scp_file}")
    else:
        print(f"Warning: RIR base directory not found: {rir_base_dir}. Skipping RIR prep.")

resamp_rir_scp_file = tmp_dir / "dns5_rirs_resampled.scp"
if resamp_rir_scp_file.exists():
    print(f"Resampled scp file already exists. Delete {resamp_rir_scp_file} if you want to re-resample.")
else:
    print("[DNS5 noise and RIR] resampling to 24kHz sampling rate")
    run_external_python_script(
        "utils/resample_to_single_fs.py",
        [
            "--in_scpfile", str(rir_scp_file),
            "--out_fs", "24000",
            "--out_scpfile", str(resamp_rir_scp_file),
            "--outdir", str(output_dir / "resampled" / "rirs"),
            "--max_files", "5000",
            "--nj", "8",
            "--chunksize", "1000"
        ],
        env_vars={"OMP_NUM_THREADS": "1"}
    )

print("Script finished successfully.")