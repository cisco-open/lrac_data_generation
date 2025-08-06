#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#set -u
#set -o pipefail

output_dir="./mls_segments"  # please do not change output_dir
CURATION_FILE="./datafiles/mls/train_meta_curated.csv"
mkdir -p "${output_dir}"

langs=("german" "french" "spanish")

echo "=== Preparing MLS data ==="

for lang in "${langs[@]}"; do
    echo "=== Preparing MLS ${lang} data ==="

    split_track="train_track1"
    split_name=$split_track

    output_dir_lang="${output_dir}/${lang}/train"
    mkdir -p "${output_dir_lang}/audio"
    if [ ! -f "${output_dir}/download_mls_${lang}.done" ]; then
        echo "[MLS-${lang}] downloading data"
        # download data from huggingface
        filelist=./datafiles/mls/mls_${lang}_${split_track}_data.txt
        cat $filelist | xargs -P 8 -I {} sh -c '
            filename=$(echo "{}" | sed -E "s|.*/([0-9]+)/([0-9]+\.tar\.gz)$|\1_\2|")
            curl -L -s -C - "https://huggingface.co/datasets/kohei0209/mls_hq_urgent_track1/resolve/main/{}?download=true" -o "$0/${filename}"
        ' "$output_dir_lang" {}
        echo "[MLS-${lang}] untarring files..."
        find "${output_dir_lang}" -name "*.tar.gz" | xargs -I {} sh -c '
            echo "  -> Extracting {}"
            tar -xzf {} -C "$0/audio"
        ' "$output_dir_lang" {}

        touch "${output_dir}/download_mls_${lang}.done"
    fi

    FULL_SCP_FILE="tmp/mls_${lang}.scp"
    ABS_OUTPUT_DIR=$(readlink -f "${output_dir}")
    if [ ! -f ${FULL_SCP_FILE} ]; then
        echo "[MLS-${lang}] creating scp file from ${ABS_OUTPUT_DIR}/${lang}/train"
        find "${ABS_OUTPUT_DIR}/${lang}/train" -type f -name "*.flac" | \
            sort | \
            awk -v lang="${lang}" '{n=split($0,a,"/"); split(a[n],b,"."); print "mls_" lang "_" b[1] " " $0}' > ${FULL_SCP_FILE}
        echo "[MLS-${lang}] created scp file with $(wc -l ${FULL_SCP_FILE} | awk '{print $1}') samples"
    else
        echo "Full dataset scp file already exists. Delete ${FULL_SCP_FILE} if you want to re-create."
    fi

    # remove low-quality samples
    FILTERED_SCP_FILE="tmp/mls_${lang}_filtered_curation.scp"
    if [ ! -f ${FILTERED_SCP_FILE} ]; then
        echo "[MLS-${lang}] filtering using curation lists"
        python utils/filter_via_curation_list.py \
            --scp_path "tmp/mls_${lang}.scp" \
            --curation_path "${CURATION_FILE}" \
            --outfile ${FILTERED_SCP_FILE}
    else
        echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
    fi

    RESAMP_SCP_FILE="tmp/mls_${lang}_resampled_filtered_curation.scp"
    if [ ! -f ${RESAMP_SCP_FILE} ]; then
        echo "[EARS] resampling to 24kHz sampling rate"
        OMP_NUM_THREADS=1 python utils/resample_to_single_fs.py \
        --in_scpfile ${FILTERED_SCP_FILE} \
        --out_fs 24000 \
        --out_scpfile ${RESAMP_SCP_FILE} \
        --outdir "${output_dir_lang}/resampled/train" \
        --max_files 5000 \
        --nj 8 \
        --chunksize 1000
    else
        echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
    fi

    echo "[MLS-${lang}] preparing data files"
    transcript_file_path=datafiles/mls/${lang}_train_transcripts.txt
    gunzip -c datafiles/mls/${lang}_train_transcripts.gz > $transcript_file_path
    sed -i "s/^/mls_${lang}_/" $transcript_file_path

    # organize the scp file
    FINAL_SCP_FILE=tmp/mls_${lang}_resampled_filtered_curation.scp
    sort -k1 $RESAMP_SCP_FILE -o $FINAL_SCP_FILE
    utils/filter_scp.pl $FINAL_SCP_FILE $transcript_file_path > tmp/mls_${lang}_resampled_filtered_curation.text
    awk '{split($1, arr, "_"); print($1" "arr[3])}' $FINAL_SCP_FILE > tmp/mls_${lang}_resampled_filtered_curation.utt2spk

    echo "[MLS-${lang}] preparing spk2gender file"
    join <(cut -d' ' -f2 tmp/mls_${lang}_resampled_filtered_curation.utt2spk | sort -u) \
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
        > tmp/mls_${lang}_resampled_filtered_curation.spk2gender
done


#--------------------------------
# Output files
# -------------------------------
# mls_${lang}_resampled_filtered_curation.scp
#    - scp file containing samples (after resampling) for training
# mls_${lang}_resampled_filtered_curation.utt2spk
#    - speaker mapping for training samples
# mls_${lang}_resampled_filtered_curation.text
#    - transcript for training samples
# mls_${lang}_resampled_filtered_curation.spk2gender
#    - speaker gender mapping for training samples