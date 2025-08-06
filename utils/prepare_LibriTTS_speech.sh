#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./libritts"
CURATION_FILE="./datafiles/libritts/train_meta_curated.csv"
mkdir -p "${output_dir}"

echo "=== Preparing LibriTTS data ==="
#################################
# Download data
#################################
# Refer to https://www.openslr.org/60/
# download in parallel using xargs
echo "Download LibriTTS data from https://www.openslr.org/60/"
urlbase="https://www.openslr.org/resources/60"
if [ ! -e "${output_dir}/download_libritts.done" ]; then
    echo "train-clean-100 train-clean-360" | tr " " "\n" \
        | xargs -n 1 -P 3 -I{} \
        wget --no-check-certificate --continue "${urlbase}/{}.tar.gz" -O "${output_dir}/{}.tar.gz"
    for x in "${output_dir}"/*.tar.gz; do
        tar xf "$x" -C "${output_dir}"
    done
else
    echo "Skip downloading and converting LibriTTS as it has already finished"
fi
touch "${output_dir}"/download_libritts.done

#################################
# Data preprocessing
#################################
mkdir -p tmp

FULL_SCP_FILE="tmp/libritts.scp"
ABS_OUTPUT_DIR=$(readlink -f "${output_dir}")
if [ ! -f ${FULL_SCP_FILE} ]; then
    echo "[LibriTTS] creating scp file from ${ABS_OUTPUT_DIR}/LibriTTS/train-clean-100 and ${ABS_OUTPUT_DIR}/LibriTTS/train-clean-360 directory"
    find "${ABS_OUTPUT_DIR}/LibriTTS/train-clean-100" -type f -name "*.wav" | \
        sort | \
        awk '{n=split($0,a,"/"); split(a[n],b,"."); gsub(/^\.\//, "", $0); print b[1] " " $0}' > ${FULL_SCP_FILE}
    find "${ABS_OUTPUT_DIR}/LibriTTS/train-clean-360" -type f -name "*.wav" | \
        sort | \
        awk '{n=split($0,a,"/"); split(a[n],b,"."); gsub(/^\.\//, "", $0); print b[1] " " $0}' >> ${FULL_SCP_FILE}
    echo "[LibriTTS] created scp file with $(wc -l ${FULL_SCP_FILE} | awk '{print $1}') samples"
else
    echo "Full dataset scp file already exists. Delete ${FULL_SCP_FILE} if you want to re-create."
fi

# remove low-quality samples
FILTERED_SCP_FILE="tmp/libritts_filtered_curation.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[LibriTTS] filtering using curation lists"
    python utils/filter_via_curation_list.py \
        --scp_path "${FULL_SCP_FILE}" \
        --curation_path "${CURATION_FILE}" \
        --outfile ${FILTERED_SCP_FILE}
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="tmp/libritts_resampled_filtered_curation.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[LibriTTS] resampling to 24kHz sampling rate"
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

echo "[LibriTTS] preparing data files"
python utils/get_libritts_transcript.py \
    --audio_scp "tmp/libritts_resampled_filtered_curation.scp" \
    --audio_dir "${output_dir}/LibriTTS/train-clean-100/" "${output_dir}/LibriTTS/train-clean-360/" \
    --outfile "tmp/libritts_resampled_filtered_curation.text" \
    --nj 8

awk '{split($1, arr, "_"); print($1" libritts_"arr[1])}' "tmp/libritts_resampled_filtered_curation.scp" > "tmp/libritts_resampled_filtered_curation.utt2spk"

echo "[LibriTTS] preparing spk2gender file"
GENDER_INFO_FILE="${output_dir}/LibriTTS/speakers.tsv"
join <(cut -d' ' -f2 "tmp/libritts_resampled_filtered_curation.utt2spk" | sort -u) \
     <(awk -F'\t' 'NR > 1 { print "libritts_" $1, tolower($2) }' "${GENDER_INFO_FILE}" | sort) \
     > "tmp/libritts_resampled_filtered_curation.spk2gender"

#--------------------------------
# Output file:
# -------------------------------
# libritts_resampled_filtered_curation.scp
#    - scp file containing resampled samples for training
# libritts_resampled_filtered_curation.utt2spk
#    - speaker mapping for filtered training samples
# libritts_resampled_filtered_curation.text
#    - transcripts for filtered training samples
# libritts_resampled_filtered_curation.spk2gender
#    - speaker gender mapping for filtered training samples