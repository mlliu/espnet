#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="wavlm_large/21" #"hmm_wavlm_1000beam_nodelta_trans_monostate" #"gmm" #"hmm_pca80_wavlm_1000beam_nodelta_trans" #_monostate" #"hmm_wavlm_nodelta_1000beam_decode_phonelm/tri3b"  #"hmm_decode_phonelm/tri4b" #"hmm_pca80_wavlm_decode_phonelm/tri4b" #"wavlm_large/21"  # use model_type/layer_index, hmm
nclusters="1000" #"2000" #"4200" #"4200" #"2500" #2000 or 2500 for hmm in forced_alignment
kmeans_cluster=true # do we use kmeans cluster as tokenizer, otherwise, we may use gmm or hmm-gmm
skip_train=false # skip the training of the model, just for decoding

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

train_set="train_nodup" #"train_clean_100"
train_dev="train_dev" #"dev"
test_sets="eval2000" #"test_clean test_other dev_clean dev_other" #"eval2000"

asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=1500 #6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

./asr2.sh \
    --kmeans_opts "--batch_bins 2400000 --nj 1" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 1 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "char" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --portion 1 \
    --inference_nj 32 \
    --kmeans_cluster ${kmeans_cluster} \
    --skip_train  ${skip_train}
    #--gpu_inference true \
