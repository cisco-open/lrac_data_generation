#!/bin/bash

# --- Shell Configuration ---
# Exit script on error (-e), on undefined variable (-u),
# on error in a pipeline (-o pipefail), and print commands (-x).
set -euo pipefail

# --- Global Variables ---
export PATH=$PATH:$PWD/utils
output_dir="./data"

# --- Helper Functions ---

# Processes a speech dataset to create Kaldi-style files (wav.scp, utt2spk, etc.).
#
# Arguments:
#   $1: dataset_name - The unique name for the dataset (e.g., "libritts").
#   $2: curation_prefix - The path and file prefix for the intermediate curation files.
#   $3: output_dir - The main directory where all data is stored.
#
process_speech_dataset() {
    local dataset_name="$1"
    local curation_prefix="$2"
    local output_dir="$3"
    local target_dir="${output_dir}/tmp/${dataset_name}"

    echo "INFO: Processing dataset: ${dataset_name}"
    mkdir -p "${target_dir}"

    # Create Kaldi-style files from the curation data
    awk '{print $1" "$3}' "${curation_prefix}.scp" > "${target_dir}/wav.scp"
    cp "${curation_prefix}.utt2spk" "${target_dir}/utt2spk"
    cp "${curation_prefix}.text" "${target_dir}/text"
    cp "${curation_prefix}.spk2gender" "${target_dir}/spk2gender"
    utils/utt2spk_to_spk2utt.pl "${target_dir}/utt2spk" > "${target_dir}/spk2utt"
    awk '{print $1" "$2}' "${curation_prefix}.scp" > "${target_dir}/utt2fs"
    awk '{print $1" 1ch_"$2"Hz"}' "${curation_prefix}.scp" > "${target_dir}/utt2category"
    cp "${target_dir}/wav.scp" "${target_dir}/spk1.scp"

    # Clean up by moving the large intermediate files
    mv "${curation_prefix}".* "${output_dir}/tmp/"
    echo "INFO: Finished processing ${dataset_name}."
}


################################################################################
#
#                                --- NOTES ---
#
# 1. Unless explicitly mentioned, no GPU is required to run these scripts.
# 2. Multiple CPUs may be required if --nj or --nsplits is specified for
#    python scripts in ./utils/.
# 3. Before running ./utils/prepare_***.sh scripts, please check the variables
#    defined at the beginning of each script and fill in appropriate values.
# 4. The `output_dir` variable specifies the directory for storing downloaded
#    audio data and metadata.
#
################################################################################


# --- Stage 1: Prepare Speech Datasets ---
echo "Stage 1: Preparing all speech datasets..."
mkdir -p "${output_dir}/tmp"

if [ ! -e "${output_dir}/tmp/dns5_librivox.done" ]; then
    echo "Preparing DNS5 LibriVox..."
    # Note: It is recommended to use GPU for get_dnsmos.py inside this script.
    ./utils/prepare_DNS5_librivox_speech.sh
    process_speech_dataset "dns5_librivox" "tmp/dns5_clean_read_speech_resampled_filtered_curation" "${output_dir}"
    touch "${output_dir}/tmp/dns5_librivox.done"
fi

if [ ! -e "${output_dir}/tmp/libritts.done" ]; then
    echo "Preparing LibriTTS..."
    ./utils/prepare_LibriTTS_speech.sh
    process_speech_dataset "libritts" "tmp/libritts_resampled_filtered_curation" "${output_dir}"
    touch "${output_dir}/tmp/libritts.done"
fi

if [ ! -e "${output_dir}/tmp/vctk.done" ]; then
    echo "Preparing VCTK..."
    ./utils/prepare_VCTK_speech.sh
    process_speech_dataset "vctk" "tmp/vctk_resampled_filtered_curation" "${output_dir}"
    touch "${output_dir}/tmp/vctk.done"
fi

if [ ! -e "${output_dir}/tmp/ears.done" ]; then
    echo "Preparing EARS..."
    ./utils/prepare_EARS_speech.sh
    process_speech_dataset "ears" "tmp/ears_resampled_filtered_curation" "${output_dir}"
    touch "${output_dir}/tmp/ears.done"
fi

if [ ! -e "${output_dir}/tmp/mls.done" ]; then
    echo "Preparing Multilingual LibriSpeech (MLS)..."
    ./utils/prepare_MLS_speech.sh
    for lang in german french spanish; do
        process_speech_dataset "mls_${lang}" "tmp/mls_${lang}_resampled_filtered_curation" "${output_dir}"
    done
    touch "${output_dir}/tmp/mls.done"
fi

if [ ! -e "${output_dir}/tmp/globe.done" ]; then
    echo "Preparing Globe..."
    python ./utils/prepare_GLOBE_speech.py
    process_speech_dataset "globe" "tmp/globe_resampled_filtered_curation" "${output_dir}"
    touch "${output_dir}/tmp/globe.done"
fi

# Combine all speech data for dynamic mixing
if [ ! -e "${output_dir}/tmp/speech_train.done" ]; then
    echo "Combining all speech datasets..."
    mkdir -p "${output_dir}/speech"
    utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" --skip_fix true "${output_dir}"/speech \
        "${output_dir}"/tmp/dns5_librivox \
        "${output_dir}"/tmp/libritts \
        "${output_dir}"/tmp/vctk \
        "${output_dir}"/tmp/ears \
        "${output_dir}"/tmp/mls_german \
        "${output_dir}"/tmp/mls_spanish \
        "${output_dir}"/tmp/mls_french \
        "${output_dir}"/tmp/globe
    python utils/flac2wav.py \
        --input_scp "${output_dir}/speech/wav.scp" \
        --num-workers 8 \
        --extra-files "text utt2spk spk2gender utt2fs utt2category spk1.scp"
    touch "${output_dir}/tmp/speech_train.done"
fi

# Sort and de-duplicate combined speech files
echo "Sorting and cleaning combined speech files..."
for f in wav.scp spk1.scp utt2spk text utt2fs utt2category spk2gender spk2utt; do
    if [ -f "${output_dir}/speech/$f" ]; then
        sort -u -o "${output_dir}/speech/$f" "${output_dir}/speech/$f"
    fi
done

# Create a validation set from the combined speech data
if [ ! -d "${output_dir}/speech_validation" ]; then
    echo "Creating speech validation set..."
    python utils/create_val_set_speech.py \
        --source_dir "${output_dir}"/speech \
        --val_dir "${output_dir}"/speech_validation \
        --min_utts 1000 \
        --max_utts_per_speaker 50
fi


# --- Stage 2: Prepare Noise and RIR Datasets ---
echo "Stage 2: Preparing all noise and RIR datasets..."
if [ ! -e "${output_dir}/tmp/dns5_noise_rir.done" ]; then
    python utils/prepare_DNS5_noise_rir.py
    touch "${output_dir}/tmp/dns5_noise_rir.done"
fi

if [ ! -e "${output_dir}/tmp/wham_noise.done" ]; then
    python utils/prepare_WHAM_noise.py
    touch "${output_dir}/tmp/wham_noise.done"
fi

if [ ! -e "${output_dir}/tmp/fsd50k_noise.done" ]; then
    python utils/prepare_FSD50K_noise.py
    touch "${output_dir}/tmp/fsd50k_noise.done"
fi

if [ ! -e "${output_dir}/tmp/fma_noise.done" ]; then
    python utils/prepare_FMA_noise.py
    touch "${output_dir}/tmp/fma_noise.done"
fi

if [ ! -e "${output_dir}/tmp/motus_rir.done" ]; then
    python utils/prepare_MOTUS_rir.py
    touch "${output_dir}/tmp/motus_rir.done"
fi

# Combine all noise and RIR data
if [ ! -e "${output_dir}/tmp/noise_rir.done" ]; then
    echo "Combining noise and RIR files..."
    cat tmp/dns5_noise_resampled_filtered_curation.scp tmp/wham_noise_resampled_filtered_curation.scp tmp/fsd50k_noise_resampled_filtered_curation.scp tmp/fma_noise_resampled_filtered_curation.scp > "${output_dir}/noise.scp"
    mv tmp/dns5_noise_resampled_filtered_curation.scp tmp/wham_noise_resampled_filtered_curation.scp tmp/fsd50k_noise_resampled_filtered_curation.scp tmp/fma_noise_resampled_filtered_curation.scp "${output_dir}/tmp/"
    python utils/flac2wav.py \
        --input_scp "${output_dir}/noise.scp" \
        --num-workers 8

    cat tmp/dns5_rirs_resampled.scp tmp/motus_rirs_resampled_filtered_curation.scp > "${output_dir}/rirs.scp"
    mv tmp/dns5_rirs_resampled.scp tmp/motus_rirs_resampled_filtered_curation.scp "${output_dir}/tmp/"
    python utils/flac2wav.py \
        --input_scp "${output_dir}/rirs.scp" \
        --num-workers 8
    touch "${output_dir}/tmp/noise_rir.done"
fi

# Create validation sets for noise and RIRs
VAL_NOISE_LIST="datafiles/validation_noise.txt"
VAL_RIR_LIST="datafiles/validation_rir.txt"
echo "Creating noise and RIR validation sets..."
for f in noise.scp rirs.scp; do
    if [ "$f" == "noise.scp" ]; then
        echo "  -> Processing noise validation set..."
        if [ -f "${VAL_NOISE_LIST}" ]; then
            python "utils/create_val_list.py" \
                --master-scp "${output_dir}/noise.scp" \
                --validation-list "${VAL_NOISE_LIST}" \
                --output-scp "${output_dir}/noise_val.scp"
        else
            echo "  Warning: Shareable noise list not found at '${VAL_NOISE_LIST}'"
        fi

    elif [ "$f" == "rirs.scp" ]; then
        echo "  -> Processing RIR validation set..."
        if [ -f "${VAL_RIR_LIST}" ]; then
            python "utils/create_val_list.py" \
                --master-scp "${output_dir}/rirs.scp" \
                --validation-list "${VAL_RIR_LIST}" \
                --output-scp "${output_dir}/rirs_val.scp"
        else
            echo "  Warning: Shareable RIR list not found at '${VAL_RIR_LIST}'"
        fi
    fi
done

echo "Data preparation script finished successfully."

################################################################################
#
#                         --- FINAL OUTPUT FILES ---
#
# ${output_dir}/
#  |- speech/
#  |   |- wav.scp, spk1.scp, utt2spk, spk2gender, spk2utt, text, utt2fs, utt2category
#  |
#  |- speech_validation/
#  |   |- wav.scp, spk1.scp, utt2spk, spk2gender, spk2utt, text, utt2fs, utt2category
#  |
#  |- noise.scp
#  |- noise_val.scp
#  |
#  |- rirs.scp
#  |- rirs_val.scp
#
################################################################################