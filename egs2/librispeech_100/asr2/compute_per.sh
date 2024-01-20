#!/usr/bin/env bash

# stage 1 convert the text to phone list using the text_to_phones.py
decode_dir="/export/fs05/mliu121/espnet_data/librispeech_100/exp/asr_train_discrete_asr_e_branchformer1_1gpu_raw_wavlm_large_21_km2000_char_ts_bpe_ts5000/decode_ctc0.3_asr_model_valid.acc.ave"
test_sets="test_clean test_other dev_clean dev_other"
for dset in ${test_sets}; do
  _scoredir="${decode_dir}/${dset}/score_wer"
  for name in hyp ref; do
    text_file="${_scoredir}/${name}.trn"
    langdir='/export/fs05/mliu121/kaldi/egs/librispeech/s5/data/lang_nosp'
    echo "convert the text to phone list for ${dset}"
    echo "${text_file}"
    python pyscripts/score/text_to_phones.py  --text_file ${text_file} --langdir ${langdir} --phone_file "${_scoredir}/${name}_phone.trn"

  done
  echo "compute the phone error rate"
  sclite \
    -r "${_scoredir}/ref_phone.trn" trn \
    -h "${_scoredir}/hyp_phone.trn" trn \
    -i rm -o all stdout > "${_scoredir}/result_phone.txt"
done

# stage 2, compute the score


