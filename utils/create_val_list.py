# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import argparse

def main():
    """
    Reconstructs a full, Kaldi-style SCP file by matching UIDs from a
    simple list against a master SCP file.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct a full SCP file from a shareable list and a master SCP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--master-scp", type=str, required=True,
        help="Path to the master SCP file containing ALL possible files (e.g., full_noise.scp)."
    )
    parser.add_argument(
        "--validation-list", type=str, required=True,
        help="Path to the simple, shareable list file (ID filename)."
    )
    parser.add_argument(
        "--output-scp", type=str, required=True,
        help="Path for the reconstructed, full validation SCP file."
    )
    args = parser.parse_args()

    print("--- Reconstructing Validation SCP ---")
    
    # Step 1: Load the master SCP into a dictionary for fast lookups.
    print(f"Reading master SCP from: {args.master_scp}")
    master_map = {}
    try:
        with open(args.master_scp, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    master_map[parts[0]] = parts[1] # {uid: "sr /path/to/file.wav"}
    except FileNotFoundError:
        print(f"ERROR: Master SCP file not found at {args.master_scp}. Aborting.")
        return
    print(f"  -> Loaded {len(master_map)} entries.")

    # Step 2: Read the validation list and reconstruct the output SCP.
    print(f"Reading validation list from: {args.validation_list}")
    lines_reconstructed = 0
    lines_missing = 0
    try:
        with open(args.validation_list, 'r', encoding='utf-8') as f_list, \
             open(args.output_scp, 'w', encoding='utf-8') as f_out:
            
            for line in f_list:
                uid = line.strip().split()[0]
                
                # Look up the UID in our master map.
                if uid in master_map:
                    # Reconstruct the full line.
                    f_out.write(f"{uid} {master_map[uid]}\n")
                    lines_reconstructed += 1
                else:
                    print(f"Warning: UID '{uid}' from validation list was not found in the master SCP. Skipping.")
                    lines_missing += 1
    except FileNotFoundError:
        print(f"ERROR: Validation list file not found at {args.validation_list}. Aborting.")
        return

    print("\n--- Reconstruction Complete ---")
    print(f"  Successfully reconstructed {lines_reconstructed} lines.")
    if lines_missing > 0:
        print(f"  Could not find {lines_missing} UIDs in the master SCP.")
    print(f"  Final SCP file saved to: {args.output_scp}")
    print("-------------------------------")

if __name__ == "__main__":
    main()