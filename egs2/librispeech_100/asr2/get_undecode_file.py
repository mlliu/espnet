# get the undecodec file
import os
# the generated decoding file is in the format of:ALFFA_amh_000202 እና ሁ ከ አስር አመት በኋላ ሸ አላሙ ድን ያችን ወጣት እዚያው የ ተዋወቁበት ዋሽን ግጠን ውስጥ በ ደማቀሰር ቅ

decode_dir = "/export/fs05/mliu121/espnet_data/challenge/exp/asr_train_discrete_asr_e_branchformer1_1gpu_lr5e-4_warmup5k_raw_hmm_wav2vec_1000beam_nodelta_trans_monostate_monophone_gaussid_km2000_char_ts_bpe_ts7000/decode_ctc0.3_asr_model_valid.acc.ave/test_1h"
decode_file = os.path.join(decode_dir, "text")

# source file
source_file = "/export/fs05/mliu121/espnet_data/challenge/data/test_1h/text"

# first