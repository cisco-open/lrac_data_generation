# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import argparse
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

def convert_worker_inplace(task):
    """
    Worker for in-place conversion: converts non-WAV files to WAV, then deletes the original.
    If the file is already a WAV, it is skipped and its path is returned.

    Args:
        task (tuple): A tuple containing (utterance_id, file_path, middle_cols).

    Returns:
        tuple or None: A tuple (utterance_id, wav_path, middle_cols) on success, None on failure.
    """
    utterance_id, path_str, middle_cols = task
    input_path = Path(path_str)
    
    if input_path.suffix.lower() == '.wav':
        return (utterance_id, str(input_path.resolve()), middle_cols)

    try:
        if not input_path.exists():
            print(f"Warning: Input file not found for utt '{utterance_id}': {input_path}")
            return None

        wav_path = input_path.with_suffix('.wav')

        command = [
            "ffmpeg", "-i", str(input_path), "-y",
            "-hide_banner", "-loglevel", "error", str(wav_path)
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        input_path.unlink()

        return (utterance_id, str(wav_path.resolve()), middle_cols)

    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {input_path}: {e}")
        return None

def main():
    """
    Main function to parse arguments and orchestrate the conversion process.
    """
    parser = argparse.ArgumentParser(
        description="""A high-efficiency tool to convert audio files from a Kaldi-style 
                       scp file to WAV. Handles N-column scp files correctly.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input_scp", type=str, required=True,
        help="Path to the input scp file (e.g., wav.scp) with audio paths."
    )
    parser.add_argument(
        "-j", "--num-workers", type=int, default=cpu_count(),
        help="Number of parallel processes to use. (default: number of CPU cores)"
    )
    parser.add_argument(
        "--extra-files", nargs='+',
        help="Space-separated list of additional files (e.g., text, utt2spk) to filter."
    )

    args = parser.parse_args()

    print(f"Reading tasks from {args.input_scp}...")
    tasks = []
    with open(args.input_scp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            if len(parts) >= 2:
                utterance_id = parts[0]
                file_path = parts[-1]
                middle_cols = parts[1:-1]
                tasks.append((utterance_id, file_path, middle_cols))
            else:
                print(f"Warning: Skipping malformed line: {line}")

    if not tasks:
        print("No valid tasks found. Exiting.")
        return

    print(f"Found {len(tasks)} files to process. Starting conversion with {args.num_workers} workers...")
    new_scp_entries = []
    with Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for result in pool.imap_unordered(convert_worker_inplace, tasks):
                if result:
                    new_scp_entries.append(result)
                pbar.update(1)

    new_scp_entries.sort(key=lambda x: x[0])
    successful_utts = {utt_id for utt_id, _, _ in new_scp_entries}

    print("\nUpdating script files...")

    temp_scp_path = Path(args.input_scp).with_suffix('.tmp')
    with open(temp_scp_path, 'w', encoding='utf-8') as f:
        for utterance_id, wav_path, middle_cols in new_scp_entries:
            line_parts = [utterance_id] + middle_cols + [wav_path]
            f.write(" ".join(line_parts) + "\n")
            
    shutil.move(str(temp_scp_path), args.input_scp)
    print(f"  -> Updated {args.input_scp}")

    if args.extra_files:
        for file_path_str in args.extra_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                print(f"Warning: Extra file not found, skipping: {file_path}")
                continue

            temp_extra_path = file_path.with_suffix('.tmp')
            lines_kept = 0
            with open(file_path, 'r', encoding='utf-8') as infile, \
                 open(temp_extra_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    if line.strip().split(maxsplit=1)[0] in successful_utts:
                        outfile.write(line)
                        lines_kept += 1
            
            shutil.move(str(temp_extra_path), str(file_path))
            print(f"  -> Updated {file_path} (kept {lines_kept} lines)")

    print("\nOperation complete!")
    print(f"Successfully processed {len(new_scp_entries)} out of {len(tasks)} files.")

if __name__ == "__main__":
    main()