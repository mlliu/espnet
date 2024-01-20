#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh

nj=1
cluster_algorithm="gmm"

kmeans_feature="wavlm_large/21"  # use model_type/layer_index
nclusters=2000 #80 for pca


train_set="train_clean_100"
train_dev="dev"
test_sets="test_clean test_other dev_clean dev_other"
train_sp_sets=

expdir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/exp
dumpdir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/dump
datadir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/data

data_audio=${dumpdir}/audio_raw
data_extract=${dumpdir}/extracted
audio_format=flac
gmm_dir="${expdir}"/gmm/$(echo "${kmeans_feature}" | tr "/" "_")_${nclusters}cluster # store the pca model
portion=0.1

kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
# TODO(simpleoier): to support features beyond s3prl
s3prl_conf="{upstream=${kmeans_feature_type}}"
kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"




    scripts/feats/perform_gmm.sh \
        --stage 2 --stop-stage 2 \
        --train_set "${train_set}" \
        --dev_set "${train_dev}" \
        --other_sets "${test_sets} ${train_sp_sets}" \
        --datadir "${data_audio}" \
        --featdir "${data_extract}" \
        --audio_format "${audio_format}" \
        --feature_type "${kmeans_feature_type}" \
        --layer "${layer}" \
        --feature_conf "${kmeans_feature_conf}" \
        --gmm_dir "${gmm_dir}" \
        --portion "${portion}" \
        --nclusters "${nclusters}" \
        --storage_save_mode false \
        --use_gpu true \
        --nj ${nj} \
        --cpu_cmd "${train_cmd}" \
        --cuda_cmd "${cuda_cmd}"
