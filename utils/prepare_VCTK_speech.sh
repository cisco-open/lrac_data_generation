#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./vctk"
CURATION_FILE="./datafiles/vctk/train_meta_curated.csv"
mkdir -p "${output_dir}"

echo "=== Preparing VCTK data ==="
#################################
# Download data
#################################
# Refer to https://datashare.ed.ac.uk/handle/10283/3443
echo "[VCTK] downloading data"
wget --continue "https://datashare.ed.ac.uk/download/DS_10283_3443.zip" -O "${output_dir}/DS_10283_3443.zip"
if [ ! -f "${output_dir}/VCTK-Corpus-0.92.zip" ]; then
    echo "Unzip DS_10283_3443.zip file"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/DS_10283_3443.zip" -d "${output_dir}"
else
    echo "Skip unzipping DS_10283_3443.zip file"
fi
if [ ! -d "${output_dir}/VCTK-Corpus" ]; then
    echo "Unzip VCTK-Corpus-0.92.zip file"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/VCTK-Corpus-0.92.zip" -d "${output_dir}/VCTK-Corpus"
else
    echo "Skip unzipping VCTK-Corpus-0.92.zip file"
fi

echo "[VCTK] preparing data files"
ABS_OUTPUT_DIR=$(readlink -f "${output_dir}")
for x in p225 p226 p227 p228 p229 p230 p231 p233 p234 p236 p237 p238 p239 p240 p241 p243 p244 p245 p246 p247 p248 p249 p250 p251 p252 p253 p254 p255 p256 p258 p259 p260 p261 p262 p263 p264 p265 p266 p267 p268 p269 p270 p271 p272 p273 p274 p275 p276 p277 p278 p279 p280 p281 p282 p283 p284 p285 p286 p287 p288 p292 p293 p294 p295 p297 p298 p299 p300 p301 p302 p303 p304 p305 p306 p307 p308 p310 p311 p312 p313 p314 p315 p316 p317 p318 p323 p326 p329 p330 p333 p334 p335 p336 p339 p340 p341 p343 p345 p347 p351 p360 p361 p362 p363 p364 p374 p376; do
    find "${ABS_OUTPUT_DIR}"/VCTK-Corpus/wav48_silence_trimmed/$x -iname '*.flac'
done | awk -F '/' '{print($NF" "$0)}' | sed -e 's/\.flac / /g' | sort -u > tmp/vctk.scp

# remove low-quality samples
FILTERED_SCP_FILE="tmp/vctk_filtered_curation.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[VCTK] filtering using curation lists"
    python utils/filter_via_curation_list.py \
        --scp_path "tmp/vctk.scp" \
        --curation_path "${CURATION_FILE}" \
        --outfile ${FILTERED_SCP_FILE}
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

RESAMP_SCP_FILE="tmp/vctk_resampled_filtered_curation.scp"
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[VCTK] resampling to 24kHz sampling rate"
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

awk '{split($1, arr, "_"); print($1" vctk_"arr[1])}' tmp/vctk_resampled_filtered_curation.scp > tmp/vctk_resampled_filtered_curation.utt2spk

python utils/get_vctk_transcript.py \
    --audio_scp tmp/vctk_resampled_filtered_curation.scp \
    --vctk_dir "${output_dir}/VCTK-Corpus" \
    --outfile tmp/vctk_resampled_filtered_curation.text \
    --nj 8

GENDER_INFO_FILE="${output_dir}/VCTK-Corpus/speaker-info.txt"
echo "[VCTK] preparing spk2gender file"
join <(cut -d' ' -f2 tmp/vctk_resampled_filtered_curation.utt2spk | sort -u) \
     <(awk 'NR > 1 {
            gender = ($3 == "Female" ? "f" : "m")
            print "vctk_" $1, gender
        }' "${GENDER_INFO_FILE}" | sort -u) \
     > tmp/vctk_resampled_filtered_curation.spk2gender

#--------------------------------
# Output files:
# -------------------------------
# vctk_resampled_filtered_curation.scp
#    - scp file containing samples for training
# vctk_resampled_filtered_curation.utt2spk
#    - speaker mapping for filtered training samples
# vctk_resampled_filtered_curation.text
#    - transcript for filtered training samples
# vctk_resampled_filtered_curation.spk2gender
#    - speaker gender mapping for filtered training samples
