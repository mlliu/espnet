# convert all the result file to the json format for submitting

import os
import json

def convert_vocab_to_json(vocab_file, target_dir):
    vocab_json = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        # vocab_json[0] = f.read().strip().split("\n")
        # read the first column of each line of the file
        vocab_json[0] = [line.strip().split()[0] for line in f]
    vocab_json_file = os.path.join(target_dir, 'vocab.json')

    json.dump(
        vocab_json,
        open(vocab_json_file, "w", encoding="utf-8"),
    )
    print("vocab json file is saved to: ", vocab_json)

def convert_token_to_json(pesuodo_data_dir, dataset, target_dir):
    for data in dataset:
        token_json = {}
        # source file
        token_file = os.path.join(pesuodo_data_dir, data, 'pseudo_labels_km5000.txt')
        # target file
        token_json_file = os.path.join(target_dir, data , 'token.json')
        if not os.path.exists( os.path.join(target_dir, data)):
            os.makedirs(os.path.join(target_dir, data))
        with open(token_file, 'r', encoding='utf-8') as token_f:
            for line in token_f:
                if len(line.strip()) == 0:
                    break
                key, value = line.strip().split(maxsplit=1)
                token_json[key] = [
                    value.split(),
                ]
            json.dump(
                token_json,
                open(token_json_file, "w", encoding="utf-8"),
            )
            # lines = f.readlines()
            # with open(token_json, 'w', encoding='utf-8') as f_new:
            #     #  {
            #     #  "AccentedFrenchOpenSLR57_fra_000005": [[784, 0, 1953, 1126, 9, 1126, 547, 273, 443, 541, 16, 768, 10, 806, 1380, 1151, 61, 382, 1004, 765, 2162, 1698, 128, 2621, 357, 914, 480, 715, 89, 1369, 893, 1307, 266, 64, 266, 681, 828, 641, 689, 1026, 488, 448, 182, 860, 1552, 628, 233, 1156, 22, 438, 659, 2239, 1125, 888, 22, 888, 1493, 752, 283, 123, 1296, 266, 64, 1000, 1187, 548, 1481, 671, 318, 629, 652, 89, 312, 1451, 88, 1826, 504, 1588, 145, 1296, 266, 64, 542, 340, 1805, 651, 217, 962, 1519, 229, 10, 403, 600]]
            #     #   }
            #     f_new.write("{\n")
            #     for line in lines:
            #         utt_id, token = line.strip().split(" ", 1)
            #         token = [int(x) for x in token.split()]
            #         f_new.write( '"' + utt_id + '": ' + "[[" + ", ".join([str(x) for x in token]) + "]]" + "\n")
            #     f_new.write("}\n")

def copy_transcript(decoding_dir, dataset, target_dir):
    for data in dataset:
        transcript_file = os.path.join(decoding_dir, data, 'text')
        target_transcript_file = os.path.join(target_dir, data, 'text')
        os.system("cp " + transcript_file + " " + target_transcript_file)

def main():
    token_type="discrete_asr_hmm_tokenizer"
    target_dir ="/export/fs05/mliu121/espnet/egs2/librispeech_100/asr2/submission/" + token_type
    #decoding_dir="/export/fs05/mliu121/espnet_data/challenge/exp/asr_train_discrete_asr_e_branchformer1_3gpu_lr5e-4_warmup5k_multi_input_raw_hmm_challenge_wavlm_monophone_gaussid_wav2vec_monophone_gaussid_km2000_char_ts_bpe_ts7000/decode_ctc0.3_asr_model_valid.acc.ave"
    #decoding_dir="/export/fs05/mliu121/espnet_data/challenge/exp/asr_train_discrete_asr_e_branchformer1_1gpu_lr1e-4_warmup75k_raw_xls_r_1b_35_km2000_bpe_rm6000_bpe_ts7000/decode_ctc0.3_asr_model_valid.acc.ave"
    decoding_dir="/export/fs05/mliu121/espnet_data/challenge/exp/asr_train_discrete_asr_e_branchformer1_2gpu_lr5e-4_warmup5k_raw_hmm_fuse_wav2vec_wavlm_monophone_gaussid_km5000_bpe_rm6000_bpe_ts7000/decode_ctc0.3_asr_model_valid.acc.ave/"
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)
    #
    # # data file and vocab file
    # # pesuodo_data_dir='/export/fs05/mliu121/espnet_data/challenge/dump/extracted/hmm_fuse_wav2vec_wavlm_monophone_gaussid/hmm_fuse_wav2vec_wavlm_monophone_gaussid'
    # pesuodo_data_dir='/export/fs05/mliu121/espnet_data/challenge/data/token_list/src_bpe_unigram6000_rm_hmm_fuse_wav2vec_wavlm_monophone_gaussid_km5000'
    # # convert the train/pseudo_labels_km5000.txt to vocabulary.txt
    # # by command: cat pseudo_labels_km5000.txt | cut -f 2- -d" "  |  tr ' ' '\n' | sort -nu > vocabulary.txt
    # #vocab_file=os.path.join(pesuodo_data_dir, 'train', 'vocabulary.txt')
    # vocab_file=os.path.join(pesuodo_data_dir, 'bpe.vocab')
    #
    #
    #
    # # first convert the vocab file to the json format
    # convert_vocab_to_json(vocab_file,target_dir)
    #
    # # convert the pseudo_labels_km5000.txt in each dataset to json format
    dataset="dev_clean dev_other test_clean test_other test_1h".split()
    # convert_token_to_json(pesuodo_data_dir, dataset, target_dir)

    # copy the transcript file to the target_dir
    copy_transcript(decoding_dir, dataset, target_dir)

if __name__ == "__main__":
    main()

