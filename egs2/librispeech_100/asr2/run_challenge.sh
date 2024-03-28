#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="hmm_fuse_wav2vec_wavlm_monophone_gaussid" #"hmm_wav2vec_1000beam_nodelta_trans_monostate_monophone_gaussid" #"hmm_wav2vec_1000beam_nodelta_trans_monostate_pdfid"  # use model_type/layer_index
nclusters=5000 #4200
kmeans_cluster=false # do we use kmeans cluster as tokenizer, otherwise, we may use gmm or hmm-g
skip_train=false  # skip the training of the model, just for decoding
skip_4a=true
skip_7a=true

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=en

dataset="challenge"
train_set="train"
train_dev="dev"
test_sets="test_clean test_other dev_clean dev_other test_1h"
dumpdir=/export/fs05/mliu121/espnet_data/challenge/dump
expdir=/export/fs05/mliu121/espnet_data/challenge/exp
datadir=/export/fs05/mliu121/espnet_data/challenge/data
#hmmdir=/export/fs05/mliu121/kaldi/egs/librispeech/s5/exp/wavlm_1000beam_nodelta_trans_monostate_monophone/mono
#hmmdir=/export/fs05/mliu121/kaldi/egs/librispeech/s5/exp/wav2vec2_1000beam_nodelta_trans_monostate/tri3b/decode_biphonelm_train/tri3b_train_decode_pdf_alignment
kaldi_root=/export/fs05/mliu121/kaldi/egs/librispeech/s5/exp/wav2vec2_1000beam_nodelta_trans_monostate/
hmmdir=${kaldi_root}/mono2/decode_phonelm_train/mono2_train_decode_gaussid

#asr_config=conf/train_discrete_asr_e_branchformer1_1gpu.yaml
#asr_config=conf/tuning/train_discrete_asr_e_branchformer1_1gpu_lr1e-4_warmup75k.yaml
#asr_config=conf/tuning/train_discrete_asr_e_branchformer1_1gpu_lr2.5e-4_warmup10k.yaml
#asr_config=conf/tuning/train_discrete_asr_e_branchformer1_1gpu_lr5e-4_warmup5k.yaml
asr_config=conf/tuning/train_discrete_asr_e_branchformer1_2gpu_lr5e-4_warmup5k.yaml
#asr_config=conf/tuning/train_discrete_asr_e_branchformer1_4gpu_lr5e-4_warmup5k.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=7000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

speed_perturb_factors=""



./asr2.sh \
    --kmeans_opts "--batch_bins 3600000" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --portion 0.3 \
    --ngpu 2 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "char" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --audio_format "flac" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --src_bpe_train_text "${datadir}/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "${datadir}/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --inference_nj 32 \
    --kmeans_cluster ${kmeans_cluster} \
    --skip_train  ${skip_train} \
    --skip_4a ${skip_4a} \
    --skip_7a ${skip_7a} \
    --dataset ${dataset} \
    --dumpdir ${dumpdir} \
    --expdir ${expdir} \
    --datadir ${datadir} \
    --hmmdir ${hmmdir}
    #--gpu_inference true
