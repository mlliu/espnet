# For wavlm feature, we only have the tokens for librispeech dataset
# so to fuse with xls_r_1b feature on challenge data, we need to pad the wavlm one to the xls_r_1b one

# the token files contains chinese characters, so we need to use utf-8 encoding
import os
import sys
import numpy as np

# 2010 ids for xls-r-1b tokens
# 2000 ids for wavlm tokens

def convert_to_int_token(lib_char_token_dir, distinct_cjk_token_lists, lib_pseudo_token_dir):
    # first build a cjk token to id mapping, open the distinct_cjk_token_lists in utf-8 encoding
    cjk_token2id = {}
    with open(distinct_cjk_token_lists, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # ech line looks like: 33 両
        for line in lines:
            token_id, token = line.strip().split()
            cjk_token2id[token] = int(token_id)
    # then we need to convert the librispeech tokens to integer tokens, and write it to the lib_pseudo_token_dir
    lib_dataset="train_clean_100 test_clean test_other dev_clean dev_other".split()
    total_int_tokens=set()
    for dataset in lib_dataset:
        count=0
        char_token_file=os.path.join(lib_char_token_dir, dataset, "text.ts.hmm_wavlm_1000beam_nodelta_trans_monostate_monophone_km2000")
        int_token_file=os.path.join(lib_pseudo_token_dir, dataset, "pseudo_labels_km2000.txt")

        if not os.path.exists(os.path.join(lib_pseudo_token_dir, dataset)):
            os.makedirs(os.path.join(lib_pseudo_token_dir, dataset))
        with open(char_token_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            with open(int_token_file, 'w', encoding='utf-8') as f_new:
                for line in lines:
                    count += 1
                    utt_id, token = line.strip().split(" ", 1)
                    # token look like:伻伻伻伻伻伻屪屪屪屪
                    # we need to convert the token to integer token
                    int_token = " ".join([str(cjk_token2id[x]) for x in token])
                    for t in int_token.split():
                        total_int_tokens.add(t)
                    f_new.write(utt_id + " " + int_token + "\n")
        print("Total number of utterances in the dataset: ", dataset, count)
    print("Total number of distinct integer tokens: ", len(total_int_tokens))




def get_librispeech_tokens(lib_pseudo_token_dir):
    # get the tokens of librispeech
    lib_utt2token={}
    lib_dataset="train_clean_100 test_clean test_other dev_clean dev_other".split()
    for dataset in lib_dataset:
        count = 0
        token_file=os.path.join(lib_pseudo_token_dir, dataset, "pseudo_labels_km2000.txt")
        with open(token_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # each line look like: 103-1240-0000 0 8 8 44 44 46 44 44 78 78 1 78 78 78 9 58 9 2 21 30 24 21 49 505 494 506
            for line in lines:
                count += 1
                utt_id, token = line.strip().split(" ", 1)
                # we need to process the token-ids, increase each number by 2010
                token = [str(int(x)+2020) for x in token.split()]
                lib_utt2token[utt_id] = " ".join(token)
        print("Total number of utterances in the dataset: ", dataset, count)
    return lib_utt2token

def gen_fuse_tokens(lib_utt2token, xls_pseudo_token_dir, new_fuse_token_dir):
    challenge_dataset="train dev test_1h dev_clean dev_other test_clean test_other".split()
    total_int_tokens = set()
    for dataset in challenge_dataset:

        total_count= 0
        lib_count = 0
        xls_token_file=os.path.join(xls_pseudo_token_dir, dataset, "pseudo_labels_km4200.txt")
        new_fuse_token_file=os.path.join(new_fuse_token_dir, dataset, "pseudo_labels_km5000.txt")
        if not os.path.exists(os.path.join(new_fuse_token_dir, dataset)):
            os.makedirs(os.path.join(new_fuse_token_dir, dataset))

        with open(xls_token_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            with open(new_fuse_token_file, 'w', encoding='utf-8') as f_new:
                for line in lines:
                    total_count += 1
                    utt_id, token = line.strip().split(" ", 1)
                    if utt_id in lib_utt2token:
                        # replace the token with the librispeech token
                        # assert the token length is the same as the xls token length
                        assert len(token.split()) == len(lib_utt2token[utt_id].split())
                        lib_token = lib_utt2token[utt_id]
                        f_new.write(utt_id + " " +lib_token + "\n")
                        lib_count += 1
                        for t in lib_token.split():
                            total_int_tokens.add(t)

                    elif utt_id[12:] in lib_utt2token:
                        assert len(token.split()) == len(lib_utt2token[utt_id[12:]].split())
                        lib_token = lib_utt2token[utt_id[12:]]
                        f_new.write(utt_id + " " + lib_token + "\n")
                        lib_count += 1
                        for t in lib_token.split():
                            total_int_tokens.add(t)
                    else:
                        f_new.write(utt_id + " " + token + "\n")
                        for t in token.split():
                            total_int_tokens.add(t)
        print("Total number of utterances in the dataset: ", dataset, total_count)
        print("librispeech tokens are used for: ", lib_count)
    print("Total number of distinct integer tokens: ", len(total_int_tokens))

def main():
    lib_char_token_dir = "/export/fs05/mliu121/espnet_data/librispeech_100_asr2/dump/raw"
    distinct_cjk_token_lists = "/export/fs05/mliu121/espnet_data/librispeech_100_asr2/exp/kmeans/distinct_cjk_token_lists"

    lib_pseudo_token_dir="/export/fs05/mliu121/espnet_data/librispeech_100_asr2/dump/extracted/hmm_wavlm_1000beam_nodelta_trans_monostate/hmm_wavlm_1000beam_nodelta_trans_monostate/"
    xls_pseudo_token_dir="/export/fs05/mliu121/espnet_data/challenge/dump/extracted/hmm_wav2vec_1000beam_nodelta_trans_monostate_monophone_gaussid/hmm_wav2vec_1000beam_nodelta_trans_monostate_monophone_gaussid/"
    new_fuse_token_dir="/export/fs05/mliu121/espnet_data/challenge/dump/extracted/hmm_fuse_wav2vec_wavlm_monophone_gaussid/hmm_fuse_wav2vec_wavlm_monophone_gaussid/"
    # before we fuse tokens, we need to convert the cjk char token to integer token based on the distinct_cjk_token_lists
    #convert_to_int_token(lib_char_token_dir, distinct_cjk_token_lists, lib_pseudo_token_dir)

    lib_utt2token = get_librispeech_tokens(lib_pseudo_token_dir)
    gen_fuse_tokens(lib_utt2token, xls_pseudo_token_dir, new_fuse_token_dir)





if __name__ == "__main__":
    main()
