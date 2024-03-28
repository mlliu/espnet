#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="hmm_wavlm_1000beam_nodelta_trans_monostate_phoneid" #"gmm" #"hmm_pca80_wavlm_1000beam_nodelta_trans" #_monostate" #"hmm_wavlm_nodelta_1000beam_decode_phonelm/tri3b"  #"hmm_decode_phonelm/tri4b" #"hmm_pca80_wavlm_decode_phonelm/tri4b" #"wavlm_large/21"  # use model_type/layer_index, hmm
nclusters="4200" #"2000" #"4200" #"4200" #"2500" #2000 or 2500 for hmm in forced_alignment
kmeans_cluster=false # do we use kmeans cluster as tokenizer, otherwise, we may use gmm or hmm-gmm
skip_train=false # skip the training of the model, just for decoding
speed_perturb= #"0.9 1.0 1.1"
src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en
skip_4a=false
skip_7a=true

dataset="librispeech_100"
#hmmdir=/export/fs05/mliu121/kaldi/egs/librispeech/s5/exp/wavlm_1000beam_nodelta_trans_monostate/tri4b_4200
hmmdir=/export/fs05/mliu121/kaldi/egs/librispeech/s5/exp/wavlm_1000beam_nodelta_trans_monostate_monophone/mono/decode_phonelm_train_clean_100/mono_train_clean_100_decode_phones_alignment
# set the data path for swbd
if [ "${dataset}" = "swbd" ]; then
    train_set="train_nodup"
    train_dev="train_dev"
    test_sets="train_dev eval2000"
    dumpdir=/export/fs05/mliu121/espnet_data/swbd/dump
    expdir=/export/fs05/mliu121/espnet_data/swbd/exp
    datadir=/export/fs05/mliu121/espnet_data/swbd/data
elif [ "${dataset}" = "librispeech_100" ]; then
    train_set="train_clean_100"
    train_dev="dev"
    test_sets="test_clean test_other dev_clean dev_other"
    dumpdir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/dump
    expdir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/exp
    datadir=/export/fs05/mliu121/espnet_data/librispeech_100_asr2/data
else
    echo "unknown dataset=${dataset}"
    exit 1
fi

# if speed_perturb is not null, then we use the asr_config for 300 hours of training data,
# in which the warmup_steps is set to 3 times of the value for 100h.
if [ -z "${speed_perturb}" ]; then
#    asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
    asr_config=conf/train_discrete_asr_e_branchformer1_1gpu_singletoken.yaml
else
    asr_config=conf/train_discrete_asr_e_branchformer1_1gpu_300h_lr1e42.yaml
fi
#asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
inference_config=conf/decode_ctc0.3.yaml
src_nbpe=6000 #1500 #6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
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
    --speed_perturb_factors "${speed_perturb}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "${datadir}/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --portion 0.1 \
    --inference_nj 32 \
    --kmeans_cluster ${kmeans_cluster} \
    --skip_train  ${skip_train} \
    --dumpdir ${dumpdir} \
    --expdir ${expdir} \
    --datadir ${datadir} \
    --skip_4a ${skip_4a} \
    --hmmdir ${hmmdir} \
    --skip_7a ${skip_7a}
#    --gpu_inference true
