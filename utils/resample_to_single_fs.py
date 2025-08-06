# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import math
from functools import partial
from pathlib import Path

import soundfile as sf
import numpy as np

# import librosa
from scipy.signal import resample_poly
from tqdm.contrib.concurrent import process_map


def resample_to_single_fs(
    uid_path_bw, idx, out_fs, max_files_per_dir, num_digits, outdir
):
    uid, audio_path = uid_path_bw

    subdir = f"{idx // max_files_per_dir:0{num_digits}x}"
    outfile = Path(outdir).resolve() / subdir / (uid + ".wav")
    # outfile = Path(outdir).resolve() / subdir / (uid + ".flac")

    if outfile.exists():
        return uid, outfile, out_fs

    try:
        audio, fs = sf.read(audio_path)
    except:
        print(f"Error: cannot open audio file '{audio_path}'. Skipping it", flush=True)
        return

    if out_fs == fs:
        return uid, audio_path, fs

    outfile.parent.mkdir(parents=True, exist_ok=True)

    divisor = np.gcd(int(fs), int(out_fs))
    p, q = int(out_fs / divisor), int(fs / divisor)
    
    audio = resample_poly(audio, p, q)
    sf.write(str(outfile), audio, out_fs)
    return uid, outfile, out_fs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_scpfile",
        type=str,
        required=True,
        help="Path to the scp file containing audio paths",
    )
    parser.add_argument(
        "--out_fs",
        type=int,
        required=True,
        help="Output sampling rate",
    )
    parser.add_argument(
        "--out_scpfile", type=str, required=True, help="Path to the output scp file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for storing resampled audios",
    )
    parser.add_argument("--nj", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--chunksize", type=int, default=1, help="Chunksize for parallel jobs"
    )
    parser.add_argument(
        "-m",
        "--max_files",
        type=int,
        default=10000,
        help="The maximum number of files per sub-directory. "
        "This is useful for systems that limit the max number of files per directory",
    )

    args = parser.parse_args()

    audios = []
    with open(args.in_scpfile, "r") as f:
        for line in f:
            uid, path = line.strip().split(maxsplit=1)
            audios.append((uid, path))

    indices = list(range(len(audios)))
    num_digits = math.ceil(math.log(len(indices) / args.max_files, 16))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ret = process_map(
        partial(
            resample_to_single_fs,
            max_files_per_dir=args.max_files,
            num_digits=num_digits,
            out_fs=args.out_fs,
            outdir=args.outdir,
        ),
        audios,
        indices,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )

    Path(args.out_scpfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_scpfile, "w") as f:
        for uid, audio_path, fs in ret:
            f.write(f"{uid} {fs} {audio_path}\n")
