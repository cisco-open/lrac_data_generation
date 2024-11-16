#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


export PATH=$PATH:$PWD/utils
output_dir="./data"
track=track1
################################################################
# Note:
#---------------------------------------------------------------
# 1. Unless explicitly mentioned, no GPU is required to run each
#    of the scripts.
# 2. Multiple CPUs may be required if the argument --nj or
#    --nsplits is specified for some python scripts in
#    ./utils/prepare_***.sh.
# 3. For the ./utils/prepare_***.sh scripts, it is recommended
#    to check the variables defined in the beginning of each
#    script and fill appropriate values before running them.
# 4. For the ./utils/prepare_***.sh scripts, the `output_dir`
#    variable is used to specify the directory for storing
#    downloaded audio data as well some meta data.
################################################################


################################
# DNSMOS models onnx files
################################
./utils/download_dnsmos_onnx.sh

################################
# Speech data
################################
mkdir -p "${output_dir}/tmp"

if [ ! -e "${output_dir}/tmp/dns5_librivox.done" ]; then
    # It is recommended to use GPU (--use_gpu True) to run `python utils/get_dnsmos.py` inside the following script
    ./utils/prepare_DNS5_librivox_speech.sh
    for subset in train; do
        mkdir -p "${output_dir}/tmp/dns5_librivox_${subset}"
        awk '{print $1" "$3}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp
        cp dns5_clean_read_speech_resampled_filtered_${subset}.utt2spk "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk
        cp dns5_clean_read_speech_resampled_filtered_${subset}.text "${output_dir}"/tmp/dns5_librivox_${subset}/text
        utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/dns5_librivox_${subset}/utt2spk > "${output_dir}"/tmp/dns5_librivox_${subset}/spk2utt
        awk '{print $1" "$2}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2fs
        awk '{print $1" 1ch_"$2"Hz"}' dns5_clean_read_speech_resampled_filtered_${subset}.scp > "${output_dir}"/tmp/dns5_librivox_${subset}/utt2category
        cp "${output_dir}"/tmp/dns5_librivox_${subset}/wav.scp "${output_dir}"/tmp/dns5_librivox_${subset}/spk1.scp
        mv dns5_clean_read_speech_resampled_filtered_${subset}.* "${output_dir}/tmp/"
    done
    mv dns5_clean_read_speech_resampled_filtered_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/dns5_librivox.done"

if [ ! -e "${output_dir}/tmp/libritts.done" ]; then
    ./utils/prepare_LibriTTS_speech.sh
    for subset in train; do
        mkdir -p "${output_dir}/tmp/libritts_${subset}"
        awk '{print $1" "$3}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/wav.scp
        cp libritts_resampled_${subset}.utt2spk "${output_dir}"/tmp/libritts_${subset}/utt2spk
        cp libritts_resampled_${subset}.text "${output_dir}"/tmp/libritts_${subset}/text
        utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/libritts_${subset}/utt2spk > "${output_dir}"/tmp/libritts_${subset}/spk2utt
        awk '{print $1" "$2}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2fs
        awk '{print $1" 1ch_"$2"Hz"}' libritts_resampled_${subset}.scp > "${output_dir}"/tmp/libritts_${subset}/utt2category
        cp "${output_dir}"/tmp/libritts_${subset}/wav.scp "${output_dir}"/tmp/libritts_${subset}/spk1.scp
        mv libritts_resampled_${subset}.* "${output_dir}/tmp/"
    done
    mv libritts_resampled_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/libritts.done"

if [ ! -e "${output_dir}/tmp/vctk.done" ]; then
    ./utils/prepare_VCTK_speech.sh
    for subset in train; do
        mkdir -p "${output_dir}/tmp/vctk_${subset}"
        awk '{print $1" "$3}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/wav.scp
        cp vctk_${subset}.utt2spk "${output_dir}"/tmp/vctk_${subset}/utt2spk
        cp vctk_${subset}.text "${output_dir}"/tmp/vctk_${subset}/text
        utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/vctk_${subset}/utt2spk > "${output_dir}"/tmp/vctk_${subset}/spk2utt
        awk '{print $1" "$2}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2fs
        awk '{print $1" 1ch_"$2"Hz"}' vctk_${subset}.scp > "${output_dir}"/tmp/vctk_${subset}/utt2category
        cp "${output_dir}"/tmp/vctk_${subset}/wav.scp "${output_dir}"/tmp/vctk_${subset}/spk1.scp
        mv vctk_${subset}.* "${output_dir}/tmp/"
    done
    mv vctk_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/vctk.done"

if [ ! -e "${output_dir}/tmp/wsj.done" ]; then
    ./utils/prepare_WSJ_speech.sh
    for subset in train; do
        mkdir -p "${output_dir}/tmp/wsj_${subset}"
        awk '{print $1" "$3}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/wav.scp
        cp wsj_${subset}.utt2spk "${output_dir}"/tmp/wsj_${subset}/utt2spk
        cp wsj_${subset}.text "${output_dir}"/tmp/wsj_${subset}/text
        utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/wsj_${subset}/utt2spk > "${output_dir}"/tmp/wsj_${subset}/spk2utt
        awk '{print $1" "$2}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2fs
        awk '{print $1" 1ch_"$2"Hz"}' wsj_${subset}.scp > "${output_dir}"/tmp/wsj_${subset}/utt2category
        cp "${output_dir}"/tmp/wsj_${subset}/wav.scp "${output_dir}"/tmp/wsj_${subset}/spk1.scp
        mv wsj_${subset}.* "${output_dir}/tmp/"
    done
    mv wsj_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/wsj.done"

if [ ! -e "${output_dir}/tmp/ears.done" ]; then
    ./utils/prepare_EARS_speech.sh
    for subset in train; do
        mkdir -p "${output_dir}/tmp/ears_${subset}"
        awk '{print $1" "$3}' ears_${subset}.scp > "${output_dir}"/tmp/ears_${subset}/wav.scp
        cp ears_${subset}.utt2spk "${output_dir}"/tmp/ears_${subset}/utt2spk
        cp ears_${subset}.text "${output_dir}"/tmp/ears_${subset}/text
        utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/ears_${subset}/utt2spk > "${output_dir}"/tmp/ears_${subset}/spk2utt
        awk '{print $1" "$2}' ears_${subset}.scp > "${output_dir}"/tmp/ears_${subset}/utt2fs
        awk '{print $1" 1ch_"$2"Hz"}' ears_${subset}.scp > "${output_dir}"/tmp/ears_${subset}/utt2category
        cp "${output_dir}"/tmp/ears_${subset}/wav.scp "${output_dir}"/tmp/ears_${subset}/spk1.scp
        mv ears_${subset}.* "${output_dir}/tmp/"
    done
    mv ears_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/ears.done"

if [ ! -e "${output_dir}/tmp/commonvoice19.done" ]; then
    ./utils/prepare_CommonVoice19_speech.sh ${track}
    for subset in train; do
        for lang in de en es fr zh-CN; do
            mkdir -p "${output_dir}/tmp/commonvoice_19_${lang}_${subset}_${track}"
            awk '{print $1" "$3}' commonvoice_19.0_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/wav.scp
            cp commonvoice_19.0_${lang}_resampled_${subset}_${track}.utt2spk "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/utt2spk
            cp commonvoice_19.0_${lang}_resampled_${subset}_${track}.text "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/text
            utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/utt2spk > "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/spk2utt
            awk '{print $1" "$2}' commonvoice_19.0_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/utt2fs
            awk '{print $1" 1ch_"$2"Hz"}' commonvoice_19.0_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/utt2category
            cp "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/wav.scp "${output_dir}"/tmp/commonvoice_19_${lang}_${subset}_${track}/spk1.scp
            mv commonvoice_19.0_${lang}_resampled_${subset}_${track}.* "${output_dir}/tmp/"
        done
    done
    mv commonvoice_19.0_*_resampled_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/commonvoice19.done"

if [ ! -e "${output_dir}/tmp/mls.done" ]; then
    ./utils/prepare_MLS_speech.sh ${track}
    for subset in train; do
        for lang in german french spanish; do
            mkdir -p "${output_dir}/tmp/mls_${lang}_${subset}_${track}"
            awk '{print $1" "$3}' mls_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/wav.scp
            cp mls_${lang}_resampled_${subset}_${track}.utt2spk "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/utt2spk
            cp mls_${lang}_resampled_${subset}_${track}.text "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/text
            utils/utt2spk_to_spk2utt.pl "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/utt2spk > "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/spk2utt
            awk '{print $1" "$2}' mls_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/utt2fs
            awk '{print $1" 1ch_"$2"Hz"}' mls_${lang}_resampled_${subset}_${track}.scp > "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/utt2category
            cp "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/wav.scp "${output_dir}"/tmp/mls_${lang}_${subset}_${track}/spk1.scp
            mv mls_${lang}_resampled_${subset}_${track}.* "${output_dir}/tmp/"
        done
    done
    mv mls_*_resampled_validation.* "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/mls.done"

# Combine all data (to be used for dynamic mixing only)
mkdir -p "${output_dir}/speech_train_${track}"
if [ ! -e "${output_dir}/tmp/speech_train_${track}.done" ]; then
    if [ $track == "track1" ]; then
        utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" --skip_fix true "${output_dir}"/speech_train_${track} \
            "${output_dir}"/tmp/dns5_librivox_train \
            "${output_dir}"/tmp/libritts_train \
            "${output_dir}"/tmp/vctk_train \
            "${output_dir}"/tmp/wsj_train \
            "${output_dir}"/tmp/ears_train \
            "${output_dir}"/tmp/commonvoice_19_de_train_${track} \
            "${output_dir}"/tmp/commonvoice_19_en_train_${track} \
            "${output_dir}"/tmp/commonvoice_19_es_train_${track} \
            "${output_dir}"/tmp/commonvoice_19_fr_train_${track} \
            "${output_dir}"/tmp/mls_german_train_${track} \
            "${output_dir}"/tmp/mls_spanish_train_${track} \
            "${output_dir}"/tmp/mls_french_train_${track}
    else
        utils/combine_data.sh --extra_files "utt2category utt2fs spk1.scp" --skip_fix true "${output_dir}"/speech_train_${track} \
        "${output_dir}"/tmp/dns5_librivox_train \
        "${output_dir}"/tmp/libritts_train \
        "${output_dir}"/tmp/vctk_train \
        "${output_dir}"/tmp/wsj_train \
        "${output_dir}"/tmp/ears_train \
        "${output_dir}"/tmp/commonvoice_19_de_train_${track} \
        "${output_dir}"/tmp/commonvoice_19_en_train_${track} \
        "${output_dir}"/tmp/commonvoice_19_es_train_${track} \
        "${output_dir}"/tmp/commonvoice_19_fr_train_${track} \
        "${output_dir}"/tmp/mls_german_train_${track} \
        "${output_dir}"/tmp/mls_english_train_${track} \
        "${output_dir}"/tmp/mls_spanish_train_${track} \
        "${output_dir}"/tmp/mls_french_train_${track}
    fi
fi
touch "${output_dir}/tmp/speech_train_${track}.done"

################################
# Noise and RIR data
################################
if [ ! -e "${output_dir}/tmp/dns5_noise_rir.done" ]; then
    ./utils/prepare_DNS5_noise_rir.sh
fi
touch "${output_dir}/tmp/dns5_noise_rir.done"

if [ ! -e "${output_dir}/tmp/wham_noise.done" ]; then
    ./utils/prepare_WHAM_noise.sh
fi
touch "${output_dir}/tmp/wham_noise.done"

if [ ! -e "${output_dir}/tmp/fsd50k_noise.done" ]; then
    ./utils/prepare_FSD50K_noise.sh
fi
touch "${output_dir}/tmp/fsd50k_noise.done"

if [ ! -e "${output_dir}/tmp/fma_noise.done" ]; then
    ./utils/prepare_FMA_noise.sh
fi
touch "${output_dir}/tmp/fma_noise.done"

if [ ! -e "${output_dir}/tmp/wind_noise.done" ]; then
    # NOTE: please change conf/wind_noise_simulation_{train,validation}.yaml if you want to change some configuration
    ./utils/prepare_wind_noise.sh
fi
touch "${output_dir}/tmp/wind_noise.done"

if [ ! -e "${output_dir}/tmp/noise_rir.done" ]; then
    # Combine all data for the training set
    cat dns5_noise_resampled_train.scp wham_noise_train.scp fsd50k_noise_resampled_train.scp fma_noise_resampled_train.scp wind_noise_train.scp > "${output_dir}/noise_train.scp"
    mv dns5_noise_resampled_train.scp wham_noise_train.scp fsd50k_noise_resampled_train.scp fma_noise_resampled_train.scp wind_noise_train.scp "${output_dir}/tmp/"
    # awk '{print $1" "$3}' ...

    # Combine all the rir data for the training set
    cat dns5_rirs.scp > "${output_dir}/rir_train.scp"
    # awk '{print $1" "$3}' ...
    mv dns5_noise_resampled_validation.scp wham_noise_validation.scp fsd50k_noise_resampled_validation.scp fma_noise_resampled_validation.scp wind_noise_validation.scp dns5_rirs.scp "${output_dir}/tmp/"
fi
touch "${output_dir}/tmp/noise_rir.done"

##########################################
# Data simulation for the validation set
##########################################
# Note: remember to modify placeholders in conf/simulation_validation.yaml before simulation.
if [ ! -e "${output_dir}/tmp/simulation_validation.done" ]; then
    utils/prepare_validation_data.sh

    mkdir -p "${output_dir}"/validation
    awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="noisy_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/validation/wav.scp 
    awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="speech_sid") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/validation/utt2spk
    utils/utt2spk_to_spk2utt.pl "${output_dir}"/validation/utt2spk > "${output_dir}"/validation/spk2utt
    awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="text") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/validation/text
    awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="clean_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/validation/spk1.scp 
    awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_validation/log/meta.tsv | sort -u -k1,1 > "${output_dir}"/validation/utt2fs
    awk '{print($1" 1ch_"$2"Hz")}' "${output_dir}"/validation/utt2fs > "${output_dir}"/validation/utt2category

    python utils/get_utt2lang.py \
        --meta_tsv simulation_validation/log/meta.tsv --outfile utt2lang
    sort -u -k1,1 utt2lang > "${output_dir}"/validation/utt2lang && rm utt2lang
fi
touch "${output_dir}/tmp/simulation_validation.done"

#######################################################
# Data simulation for a fixed training set (optional)
#######################################################
# mkdir -p simulation_train/log
# if [ ! -e "${output_dir}/tmp/simulation_train.done" ]; then
#     python simulation/generate_data_param.py --config conf/simulation_train.yaml
#     # It takes ~1.5 hours to finish simulation with nj=8
#     OMP_NUM_THREADS=1 python simulation/simulate_data_from_param.py \
#         --config conf/simulation_train.yaml \
#         --meta_tsv simulation_train/log/meta.tsv \
#         --nj 8 \
#         --chunksize 200
# fi
# touch "${output_dir}/tmp/simulation_train.done"

# mkdir -p "${output_dir}"/train
# awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="noisy_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u > "${output_dir}"/train/wav.scp 
# awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="speech_sid") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u > "${output_dir}"/train/utt2spk
# utils/utt2spk_to_spk2utt.pl "${output_dir}"/train/utt2spk > "${output_dir}"/train/spk2utt
# awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="text") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u > "${output_dir}"/train/text
# awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="clean_path") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u > "${output_dir}"/train/spk1.scp 
# awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' simulation_train/log/meta.tsv | sort -u > "${output_dir}"/train/utt2fs
# awk '{print($1" 1ch_"$2"Hz")}' "${output_dir}"/train/utt2fs > "${output_dir}"/train/utt2category

#--------------------------------
# Output files:
# -------------------------------
# ${output_dir}/speech_train/
#  |- wav.scp
#  |- spk1.scp
#  |- utt2spk
#  |- spk2utt
#  |- utt2fs
#  \- utt2category
#
# ${output_dir}/validation/
#  |- wav.scp
#  |- spk1.scp
#  |- utt2spk
#  |- spk2utt
#  |- text
#  |- utt2fs
#  \- utt2category
#
# ${output_dir}/noise_train.scp
#
# ${output_dir}/rir_train.scp
