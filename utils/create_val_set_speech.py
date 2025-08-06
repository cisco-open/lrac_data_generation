# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import random
import shutil
import argparse
from itertools import zip_longest

def create_validation_set(source_dir, val_dir, min_utts=1000, max_utts_per_speaker=None, balance_gender=True, random_seed=42):
    """
    Creates a validation set from a Kaldi-style data directory by selecting
    random speakers.

    This script ensures that all utterances from a selected speaker are moved
    to the validation set, preventing speaker overlap between train and val sets.
    It supports deterministic behavior via a random seed and can balance the
    validation set by gender if a 'spk2gender' file is available.

    Args:
        source_dir (str): The path to the original data directory (e.g., 'data/train').
        val_dir (str): The path to the new validation directory to be created (e.g., 'data/val').
        min_utts (int): The minimum number of utterances desired for the validation set.
        max_utts_per_speaker (int, optional): If provided, only speakers with at most
                                              this many utterances will be considered for
                                              the validation set. Defaults to None.
        balance_gender (bool): If True, attempts to create a gender-balanced validation set.
                               Requires a 'spk2gender' file in the source directory.
        random_seed (int): Seed for the random number generator to ensure deterministic splits.
    """
    random.seed(random_seed)

    print(f"Source directory: {source_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Minimum validation utterances: {min_utts}")
    print(f"Random seed: {random_seed}")
    if max_utts_per_speaker is not None:
        print(f"Maximum utterances per speaker for validation eligibility: {max_utts_per_speaker}")

    # --- 1. Validate paths and files ---
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    required_files = ['spk2utt', 'utt2spk', 'wav.scp', 'text']
    for f in required_files:
        if not os.path.exists(os.path.join(source_dir, f)):
            print(f"Error: Required file '{f}' not found in '{source_dir}'.")
            return

    # List of all files to be split
    kaldi_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    print(f"Found files to process: {', '.join(kaldi_files)}")


    # --- 2. Read spk2utt to get speaker and utterance info ---
    print("\nReading speaker and utterance information from spk2utt...")
    spk2utt_path = os.path.join(source_dir, 'spk2utt')
    speaker_to_utts = {}
    speaker_utt_counts = {}
    all_speakers = []

    with open(spk2utt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            utterances = parts[1:]
            speaker_to_utts[speaker_id] = utterances
            speaker_utt_counts[speaker_id] = len(utterances)
            all_speakers.append(speaker_id)

    print(f"Found {len(all_speakers)} speakers in total.")

    # --- Read spk2gender if gender balancing is set ---
    speaker_to_gender = {}
    spk2gender_path = os.path.join(source_dir, 'spk2gender')
    if balance_gender:
        if not os.path.exists(spk2gender_path):
            print(f"\nWarning: --balance_gender was specified, but '{spk2gender_path}' not found.")
            print("Falling back to non-gender-balanced selection.")
            balance_gender = False
        else:
            print("\nReading spk2gender for gender balancing...")
            with open(spk2gender_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    speaker_to_gender[parts[0]] = parts[1]

    # --- 3. Filter speakers based on max_utts_per_speaker ---
    print(f"\nFiltering speakers...")
    eligible_speakers = [
        spk for spk in all_speakers
        if max_utts_per_speaker is None or speaker_utt_counts[spk] <= max_utts_per_speaker
    ]
    if not eligible_speakers:
        print("Error: No speakers match the specified criteria (e.g., --max_utts_per_speaker).")
        return
    print(f"Found {len(eligible_speakers)} eligible speakers for validation set.")

    # --- 4. Select speakers for the validation set ---
    validation_speakers = set()
    validation_utterances = set()
    current_utt_count = 0

    if balance_gender:
        print("\nAttempting gender-balanced selection...")
        eligible_males = [spk for spk in eligible_speakers if speaker_to_gender.get(spk, 'f') == 'm']
        eligible_females = [spk for spk in eligible_speakers if speaker_to_gender.get(spk, 'm') == 'f']
        random.shuffle(eligible_males)
        random.shuffle(eligible_females)
        print(f"Found {len(eligible_males)} eligible male and {len(eligible_females)} eligible female speakers.")

        # Alternate between adding male and female speakers
        for male_spk, female_spk in zip_longest(eligible_males, eligible_females):
            if current_utt_count >= min_utts:
                break
            # Add a male speaker if available
            if male_spk and male_spk not in validation_speakers:
                validation_speakers.add(male_spk)
                validation_utterances.update(speaker_to_utts[male_spk])
                current_utt_count += speaker_utt_counts[male_spk]
            # Add a female speaker if available
            if female_spk and female_spk not in validation_speakers:
                validation_speakers.add(female_spk)
                validation_utterances.update(speaker_to_utts[female_spk])
                current_utt_count += speaker_utt_counts[female_spk]
    else:
        print(f"\nSelecting speakers randomly for validation set (target size >= {min_utts} utterances)...")
        random.shuffle(eligible_speakers)
        for speaker in eligible_speakers:
            if current_utt_count >= min_utts:
                break
            validation_speakers.add(speaker)
            validation_utterances.update(speaker_to_utts[speaker])
            current_utt_count += speaker_utt_counts[speaker]

    if not validation_speakers:
        print("Error: Could not select any speakers. Check dataset size and target utterance count.")
        return

    training_speakers = set(all_speakers) - validation_speakers

    print(f"Selected {len(validation_speakers)} speakers for the validation set.")
    print(f"Total utterances in validation set: {len(validation_utterances)}")
    print(f"Remaining speakers in training set: {len(training_speakers)}")

    # --- 5. Create validation directory and split files ---
    print(f"\nCreating validation directory at '{val_dir}' and splitting files...")
    os.makedirs(val_dir, exist_ok=True)

    # Files keyed by utterance ID vs. speaker ID
    utt_keyed_files = ['text', 'utt2category', 'utt2fs', 'utt2spk', 'wav.scp']
    spk_keyed_files = ['spk2utt', 'spk2gender', 'spk1.csp'] # Add any other speaker-keyed files here

    for filename in kaldi_files:
        print(f"  Processing '{filename}'...")
        original_path = os.path.join(source_dir, filename)
        train_path_tmp = os.path.join(source_dir, f"{filename}.tmp")
        val_path = os.path.join(val_dir, filename)

        # Identify if a file is speaker-keyed, utterance-keyed, or unknown
        file_base = os.path.basename(filename)
        key_type = 'unknown'
        if file_base in utt_keyed_files:
            key_type = 'utt'
        elif file_base in spk_keyed_files:
            key_type = 'spk'

        with open(original_path, 'r', encoding='utf-8') as f_in, \
             open(train_path_tmp, 'w', encoding='utf-8') as f_train, \
             open(val_path, 'w', encoding='utf-8') as f_val:

            for line in f_in:
                key = line.strip().split(maxsplit=1)[0]
                is_in_val = False
                if key_type == 'utt':
                    if key in validation_utterances: is_in_val = True
                elif key_type == 'spk':
                    if key in validation_speakers: is_in_val = True
                else: # Fallback for unknown files: check both sets
                    if key in validation_utterances or key in validation_speakers:
                        is_in_val = True

                # Write to the appropriate file
                if is_in_val:
                    f_val.write(line)
                else:
                    f_train.write(line)

    # --- 6. Replace original files with the new, smaller training files ---
    print("\nFinalizing training set files...")
    for filename in kaldi_files:
        original_path = os.path.join(source_dir, filename)
        train_path_tmp = os.path.join(source_dir, f"{filename}.tmp")
        # Overwrite the original file with the temporary file
        shutil.move(train_path_tmp, original_path)

    print("\nValidation set creation complete!")
    print(f"Training data is updated in: {source_dir}")
    print(f"Validation data has been created in: {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a speaker-disjoint validation set from a Kaldi-style data directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Path to the original data directory (e.g., data/train)."
    )
    parser.add_argument(
        "--val_dir", type=str, required=True,
        help="Path to the new validation directory to be created (e.g., data/val)."
    )
    parser.add_argument(
        "--min_utts", type=int, default=1000,
        help="The minimum number of utterances for the validation set."
    )
    parser.add_argument(
        "--max_utts_per_speaker", type=int, default=None,
        help="If specified, only select speakers with at most this many utterances for the validation set."
    )
    parser.add_argument(
        "--no_balance_gender", action='store_false',
        help="Do not attempt to balance the validation set by gender."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for deterministic speaker selection."
    )

    args = parser.parse_args()
    create_validation_set(
        args.source_dir,
        args.val_dir,
        args.min_utts,
        args.max_utts_per_speaker,
        args.no_balance_gender,
        args.random_seed
    )