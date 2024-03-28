
token_suffix="hmm_fuse_wav2vec_wavlm_monophone_gaussid_km5000"
bpe_folder="data/token_list/src_bpe_unigram6000_rm_hmm_fuse_wav2vec_wavlm_monophone_gaussid_km5000"
bitrate_dir="./bitrate"
for dset in dev_clean dev_other test_clean test_other test_1h; do
paste \
     <(<dump/raw/${dset}/text.rm.${token_suffix} cut -d" " -f1) \
     <(<dump/raw/${dset}/text.rm.${token_suffix} spm_encode --model=${bpe_folder}/bpe.model --output_format=id) \
     > dump/raw/${dset}/token_int.rm.${token_suffix}

   python pyscripts/utils/convert_token2json.py \
     --vocab data/token_list/src_bpe_unigram6000_rm_${token_suffix}/tokens.txt \
     --token dump/raw/${dset}/token_int.rm.${token_suffix} \
     --ref_scp data/${dset}/wav.scp \
     --result_dir "${bitrate_dir}/${dset}"

   python pyscripts/utils/calculate_bitrate.py \
     --vocab "${bitrate_dir}/${dset}"/vocab.json \
     --tokens "${bitrate_dir}/${dset}"/tokens.json \
     --reference_len "${bitrate_dir}/${dset}"/ref_len.scp \
     --bitrate_details "${bitrate_dir}/${dset}"/details.txt
 done

# python - <<EOF
#import numpy as np
#bitrates=[]
#bitrate_dir="${bitrate_dir}"
#for dset in ["dev_clean", "dev_other", "test_clean", "test_other", "test_1h"]:
#    with open(f"{bitrate_dir}/{dset}/details.txt", "r") as f:
#        for line in f.readlines():
#            lst = line.strip().split()
#            bitrates.append(float(lst[1]))
#print(np.round(np.mean(bitrates), 2))