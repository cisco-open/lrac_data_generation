#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#set -u
#set -o pipefail

output_dir="./ears"
CURATION_FILE="./datafiles/ears/train_meta_curated.csv"
mkdir -p "${output_dir}"

echo "=== Preparing EARS data ==="
#################################
# Download data
#################################

echo "Download EARS data"
if [ ! -e "${output_dir}/download_ears.done" ]; then
    seq -w 001 101 | xargs -I {} -P 8 bash -c '
        X="{}"
        output_dir="$1"
        echo "Downloading EARS ${output_dir}/p${X}.zip"
        curl -s -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -o "${output_dir}/p${X}.zip"
        unzip -q "${output_dir}/p${X}.zip" -d "$output_dir"
        rm "${output_dir}/p${X}.zip"
    ' _ "$output_dir"

    git clone https://github.com/facebookresearch/ears_dataset.git "${output_dir}/ears_scripts"
else
    echo "Skip downloading EARS as it has already finished"
fi
touch "${output_dir}"/download_ears.done

echo "[EARS] preparing data files"

# Train data
ABS_OUTPUT_DIR=$(readlink -f "${output_dir}")
for x in $(seq 1 101); do
    xx=$(printf "%03d" "$x")
    find "${ABS_OUTPUT_DIR}/p$xx" -iname '*.wav'
done | awk -F '/' '{print($(NF-1)"_"$NF" "$0)}' | sed -e 's/\.wav / /g' | sort -u > tmp/ears.scp

# remove low-quality samples
FILTERED_SCP_FILE="tmp/ears_filtered_curation.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[EARS] filtering using curation lists"
    python utils/filter_via_curation_list.py \
        --scp_path "tmp/ears.scp" \
        --curation_path "${CURATION_FILE}" \
        --outfile ${FILTERED_SCP_FILE}
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="tmp/ears_resampled_filtered_curation.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[EARS] resampling to 24kHz sampling rate"
    OMP_NUM_THREADS=1 python utils/resample_to_single_fs.py \
       --in_scpfile ${FILTERED_SCP_FILE} \
       --out_fs 24000 \
       --out_scpfile ${RESAMP_SCP_FILE} \
       --outdir "${output_dir}/resampled/train" \
       --max_files 5000 \
       --nj 8 \
       --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

# utt2spk
awk '{split($1, arr, "_"); print($1" ears_"arr[1])}' tmp/ears_resampled_filtered_curation.scp > tmp/ears_resampled_filtered_curation.utt2spk

# spk2gender
echo "[EARS] creating spk2gender file using jq"
jq -r '
  to_entries[] |
  .key as $p_id |
  .value.gender as $gender |
  "ears_\($p_id) \(
    if $gender == "male" then "m"
    elif $gender == "female" then "f"
    else "o"
    end
  )"
' "${output_dir}/ears_scripts/speaker_statistics.json" | sort > tmp/ears.spk2gender.full

echo "[EARS] filtering spk2gender list to match the final speaker set"
cut -d' ' -f2 tmp/ears_resampled_filtered_curation.utt2spk | sort -u > tmp/speakers.list
grep -f tmp/speakers.list tmp/ears.spk2gender.full > tmp/ears_resampled_filtered_curation.spk2gender
rm tmp/ears.spk2gender.full tmp/speakers.list

# transcription
python utils/get_ears_transcript.py \
    --audio_scp tmp/ears_resampled_filtered_curation.scp \
    --transcript_json_path "${output_dir}/ears_scripts/transcripts.json" \
    --outfile tmp/ears_resampled_filtered_curation.text


#--------------------------------
# Output file:
# -------------------------------
# ears_resampled_filtered_curation.scp
#    - scp file for training
# ears_resampled_filtered_curation.utt2spk
#    - speaker mapping for training samples
# ears_resampled_filtered_curation.text
#    - transcripts for training samples
# ears_resampled_filtered_curation.spk2gender
#    - speaker to gender mapping for training samples