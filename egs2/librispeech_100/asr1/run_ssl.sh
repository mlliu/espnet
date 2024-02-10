#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_SSL_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type ssl \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --feats_normalize uttmvn \
    --stage 12 \
    --stop_stage 13 \
    --skip_stages 5   # skip stage 5 (stage 5 is for bpe training
    #--asr_stats_dir "/export/fs05/mliu121/espnet_data/librispeech_100_asr1/exp/asr_stats_raw_en_bpe5000_SSL"
    #--speed_perturb_factors "0.9 1.0 1.1" \
