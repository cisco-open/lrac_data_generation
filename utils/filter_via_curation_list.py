# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse


def filter_curation_list(info, curation_list, exclude=False):
    """
    Filters a list of audio samples based on a curation list.

    Args:
        info (dict): A dictionary mapping UIDs to audio file paths.
        curation_list (pd.DataFrame): A DataFrame with a 'uid' or 'filename' column.
        exclude (bool): If True, excludes samples in the curation list.
                        If False, includes only samples in the curation list.

    Returns:
        list: A list of UIDs that pass the filter.
    """
    # Create sets of curated items for efficient lookup.
    has_uid = 'uid' in curation_list.columns
    curated_uids = set()
    if has_uid:
        curated_uids = set(curation_list['uid'])

    has_filename = 'filename' in curation_list.columns
    curated_filenames = set()
    if has_filename:
        curated_filenames = set(curation_list['filename'])

    if not curated_uids and not curated_filenames:
        raise ValueError("Curation list must contain either a 'uid' or 'filename' column.")

    filtered_uids = []
    # Iterate over the main list of samples (info)
    for uid, path in tqdm(info.items(), desc="Filtering samples"):
        filename = Path(path).name

        # Check if the sample is in the curation list by either uid or filename
        if has_uid:
            is_curated = uid in curated_uids
        elif has_filename:
            is_curated = filename in curated_filenames
        else:
            raise ValueError("Curation list must contain either a 'uid' or 'filename' column.")

        # If in exclude mode, keep the sample if it's NOT curated.
        # If in include mode (default), keep the sample if it IS curated.
        if exclude:
            if not is_curated:
                filtered_uids.append(uid)
        else:
            if is_curated:
                filtered_uids.append(uid)
                
    return filtered_uids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter an audio scp file based on a curation list.")
    parser.add_argument(
        "--scp_path",
        type=str,
        required=True,
        help="Path to the scp file containing audios",
    )
    parser.add_argument(
        "--curation_path",
        type=str,
        required=True,
        help="Path to the curation list file (CSV format)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output file for storing filtered samples",
    )
    parser.add_argument(
        "--exclude",
        action="store_true",
        help="Exclude files from the curation list instead of including them.",
    )
    args = parser.parse_args()

    curation_list = pd.read_csv(args.curation_path, low_memory=False)

    info = {}
    with open(args.scp_path, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            info[uid] = audio_path
    print(f"Loaded {len(info)} unique samples from {args.scp_path}")
    
    uids = filter_curation_list(info, curation_list, exclude=args.exclude)
    
    mode_text = "Excluding" if args.exclude else "Including"
    print(f"Filtering mode: {mode_text}. Based on {len(curation_list)} curation entries: {len(info)} samples -> {len(uids)} samples")

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        for uid in uids:
            if uid in info:
                f.write(f"{uid} {info[uid]}\n")
    
    print(f"Filtered list saved to {args.outfile}")