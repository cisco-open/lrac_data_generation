#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./dns5_fullband" # please do not change this path
CURATION_FILE="./datafiles/dns5_librivox_speech/train_meta_curated.csv"

#################################
# Download data
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
./utils/download_librivox_speech.sh ${output_dir} 8

#################################
# Data preprocessing
#################################
mkdir -p tmp
FULL_SCP_FILE="tmp/dns5_clean_read_speech.scp"
ABS_OUTPUT_DIR=$(readlink -f "${output_dir}")
if [ ! -f ${FULL_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] creating scp file from ${ABS_OUTPUT_DIR}/Track1_Headset/mnt directory"
    find "${ABS_OUTPUT_DIR}/Track1_Headset/mnt" -type f -name "*.wav" | \
        sort | \
        awk '{
            n=split($0,a,"/");
            split(a[n],b,".");
            key = b[1];
            if (seen[key]++) {
                key = key "(" seen[key] ")"
            }
            print key " " $0
        }' > "${FULL_SCP_FILE}"
    echo "[DNS5 LibriVox] created scp file with $(wc -l ${FULL_SCP_FILE} | awk '{print $1}') samples"
else
    echo "Full dataset scp file already exists. Delete ${FULL_SCP_FILE} if you want to re-create."
fi

# remove low-quality samples
FILTERED_SCP_FILE="tmp/dns5_clean_read_speech_filtered_curation.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] filtering using curation lists"
    python utils/filter_via_curation_list.py \
        --scp_path "${FULL_SCP_FILE}" \
        --curation_path "${CURATION_FILE}" \
        --outfile ${FILTERED_SCP_FILE}
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="tmp/dns5_clean_read_speech_resampled_filtered_curation.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] resampling to 24kHz sampling rate"
    OMP_NUM_THREADS=1 python utils/resample_to_single_fs.py \
       --in_scpfile ${FILTERED_SCP_FILE} \
       --out_fs 24000 \
       --out_scpfile ${RESAMP_SCP_FILE} \
       --outdir "${output_dir}/Track1_Headset/resampled/clean/read_speech" \
       --max_files 5000 \
       --nj 8 \
       --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

echo "[DNS5 LibriVox] preparing data files"
sort -u tmp/dns5_clean_read_speech_resampled_filtered_curation.scp | \
    awk '{split($1, arr, "_"); if(arr[5]!="reader"){exit 1;} spk=arr[5]"_"arr[6]; print($1" dns5_"spk)}' > tmp/dns5_clean_read_speech_resampled_filtered_curation.utt2spk

awk '{print($1" <not-available>")}' tmp/dns5_clean_read_speech_resampled_filtered_curation.scp > tmp/dns5_clean_read_speech_resampled_filtered_curation.text

echo "[DNS5 LibriVox] preparing spk2gender file"
join <(cut -d' ' -f2 tmp/dns5_clean_read_speech_resampled_filtered_curation.utt2spk | sort -u) \
     <(awk -F, '
        NR > 1 {
            speaker_id = $(NF-1)
            gender_str = $(NF)
            gsub(/\r|"/, "", gender_str)
            gender = (gender_str == "male" ? "m" : "f")
            if (speaker_id != "") {
                print speaker_id, gender
            }
        }' "${CURATION_FILE}" | sort -u) \
     > tmp/dns5_clean_read_speech_resampled_filtered_curation.spk2gender

#--------------------------------
# Output file:
# -------------------------------
# dns5_clean_read_speech_resampled_filtered_curation.scp
#    - scp file containing filtered samples (after resampling) for training
# dns5_clean_read_speech_resampled_filtered_curation.utt2spk
#    - speaker mapping for filtered training samples
# dns5_clean_read_speech_resampled_filtered_curation.text
#    - transcript for filtered training samples
# dns5_clean_read_speech_resampled_filtered_curation.spk2gender
#    - speaker gender mapping for filtered training samples
