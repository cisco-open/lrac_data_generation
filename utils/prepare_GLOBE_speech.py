# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from tqdm import tqdm
import io
import dask.dataframe as dd
import soundfile as sf
from pathlib import Path
import requests
import pandas as pd
import threading
import concurrent.futures


# Download parquet files 0000 to 0107 directly from URL
base_url = "https://huggingface.co/datasets/MushanW/globe/resolve/refs%2Fconvert%2Fparquet/default/train/"
cache_dir = Path("./globe/cache")
cache_dir.mkdir(parents=True, exist_ok=True)

parquet_files = []
parquet_files_lock = threading.Lock()

def download_file(i):
    file_num = f"{i:04d}"
    url = f"{base_url}{file_num}.parquet?download=true"
    parquet_path = cache_dir / f"globe_train_{file_num}.parquet"

    if parquet_path.exists():
        print(f"File {file_num}.parquet already exists, skipping download.")
        with parquet_files_lock:
            parquet_files.append(str(parquet_path))
        return

    print(f"Downloading {file_num}.parquet...")
    response = requests.get(url)

    with open(parquet_path, "wb") as f:
        f.write(response.content)

    with parquet_files_lock:
        parquet_files.append(str(parquet_path))

# Download files with max 8 concurrent downloads
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(download_file, range(108))

# Combine all parquet files into one Dask dataframe
df = dd.read_parquet(parquet_files)

# Create output directories
output_dir = Path("./globe")
split_dir = output_dir / "train"
split_dir.mkdir(parents=True, exist_ok=True)
Path("./tmp").mkdir(parents=True, exist_ok=True)

# Check if all output files already exist
output_files = [
    "tmp/globe.scp",
    "tmp/globe_filtered_curation.scp",
    "tmp/globe_resampled_filtered_curation.scp",
    "tmp/globe_resampled_filtered_curation.utt2spk",
    "tmp/globe_resampled_filtered_curation.text",
    "tmp/globe_resampled_filtered_curation.spk2gender"
]

if all(Path(f).exists() for f in output_files):
    print("All output files already exist, skipping processing.")
    exit(0)

# Initialize Kaldi files
globe_scp = open("tmp/globe.scp", "w")
globe_scp_filtered = open("tmp/globe_filtered_curation.scp", "w")
globe_scp_resampled = open("tmp/globe_resampled_filtered_curation.scp", "w")
utt2spk = open("tmp/globe_resampled_filtered_curation.utt2spk", "w")
text_file = open("tmp/globe_resampled_filtered_curation.text", "w")
unique_spk2gender = set()

metadata= pd.read_csv('./datafiles/globe/train_meta_curated.csv', low_memory=False)
print(f"Loaded metadata with {len(metadata)} samples.")

curated_pairs = set(zip(metadata['filename'], metadata['speaker_id']))
print(f"Created a lookup set with {len(curated_pairs)} curated (filename, speaker_id) pairs.")

# Process data in chunks to handle large datasets using 8 threads
def process_sample(args):
    i, row = args
    _, sample = row

    utt_id = sample['audio']['path'].split(".")[0]
    spk_id = sample.get("speaker_id")

    # Create speaker-specific directory
    spk_dir = split_dir / "flac" / spk_id
    spk_dir.mkdir(parents=True, exist_ok=True)

    # Write audio file
    audio_path = spk_dir / f"{utt_id}.flac"
    file_like = io.BytesIO(sample['audio']['bytes'])
    audio, sample_rate = sf.read(file_like)
    sf.write(str(audio_path), audio, sample_rate)

    # Extract filename from audio_path
    filename = audio_path.name

    results = {
        'globe_scp': f"{spk_id}_{utt_id} {audio_path.absolute()}\n",
        'globe_scp_filtered': None,
        'globe_scp_resampled': None,
        'utt2spk': None,
        'text': None,
        'spk2gender': None
    }

    # Check if filename is present in metadata
    if (filename, spk_id) in curated_pairs:
        results['globe_scp_filtered'] = f"{spk_id}_{utt_id} {audio_path.absolute()}\n"
        results['globe_scp_resampled'] = f"{spk_id}_{utt_id} 24000 {audio_path.absolute()}\n"
        results['utt2spk'] = f"{spk_id}_{utt_id} globe_{spk_id}\n"

        # Write text if available
        if "transcript" in sample:
            results['text'] = f"{spk_id}_{utt_id} {sample['transcript']}\n"
        else:
            results['text'] = f"{spk_id}_{utt_id} <not-available>\n"

        # Get gender from metadata
        gender = sample.get('gender', 'o')[0].lower()
        results['spk2gender'] = (f"globe_{spk_id}", gender)

    return results

# Process with ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Submit all tasks
    futures = [executor.submit(process_sample, (i, row)) for i, row in enumerate(df.iterrows())]

    # Collect results with progress bar
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()

        # Write results to files (this part remains single-threaded to avoid file conflicts)
        globe_scp.write(result['globe_scp'])

        if result['globe_scp_filtered']:
            globe_scp_filtered.write(result['globe_scp_filtered'])
        if result['globe_scp_resampled']:
            globe_scp_resampled.write(result['globe_scp_resampled'])
        if result['utt2spk']:
            utt2spk.write(result['utt2spk'])
        if result['text']:
            text_file.write(result['text'])
        if result['spk2gender']:
            unique_spk2gender.add(result['spk2gender'])

globe_scp.close()
globe_scp_filtered.close()
globe_scp_resampled.close()
utt2spk.close()
text_file.close()

with open("tmp/globe_resampled_filtered_curation.spk2gender", "w") as f:
    # Sort the list for a consistent, deterministic output file
    for spk, gen in sorted(list(unique_spk2gender)):
        f.write(f"{spk} {gen}\n")