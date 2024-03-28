# For wavlm feature, we only have the tokens for librispeech dataset
# so to fuse with xls_r_1b feature on challenge data, we need to pad the wavlm one to the xls_r_1b one

# the token files contains chinese characters, so we need to use utf-8 encoding
import os
import sys
import numpy as np

def main():
    # distinct_cjk_token_file, use the last token of the file as the padding token
    # make sure that this token is not in the wavlm token list
    distinct_cjk_token_file = "/export/fs05/mliu121/espnet_data/librispeech_100_asr2/exp/kmeans/distinct_cjk_token_lists"

    #wavlm_token="hmm_wavlm_1000beam_nodelta_trans_monostate_tgmed_km2000"
    wavlm_token="hmm_wavlm_1000beam_nodelta_trans_monostate_monophone_km2000"
    wavlm_lib_token = "text.ts." + wavlm_token
    wavlm_lib_dir = "/export/fs05/mliu121/espnet_data/librispeech_100_asr2/dump/raw"
    lib_dataset = ["train_clean_100", "dev_clean", "dev_other", "test_clean", "test_other"]
    wavlm_token_listdir= "/export/fs05/mliu121/espnet_data/librispeech_100_asr2/data/token_list"
    token_list_name = "char_" + wavlm_token


    wav2vec_challenge_token = "text.ts.hmm_wav2vec_1000beam_nodelta_trans_monostate_pdfid_km4200"
    wav2vec_challenge_dir = "/export/fs05/mliu121/espnet_data/challenge/dump/raw"
    challenge_dataset =["train", "dev", "test_clean", "test_other", "dev_clean", "dev_other","test_1h"]
    challenge_token_listdir = "/export/fs05/mliu121/espnet_data/challenge/data/token_list"

    # first iterate through all the librispeech dataset, collect a dictionary of all the utterance ids and their corresponding tokens sequence
    lib_utt2token = {}
    for dataset in lib_dataset:
        token_file = os.path.join(wavlm_lib_dir, dataset, wavlm_lib_token)
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, token_seq = line.strip().split(" ", 1)
                lib_utt2token[utt_id] = token_seq

    # open the distinct_cjk_token_file, and read the last token
    # each line in the file is an index and a token: 0 一
    with open(distinct_cjk_token_file, "r", encoding="utf-8") as f:
        for line in f:
            pass
        padding_token = line.strip().split()[1]
    # assert that the padding token is not in the wavlm token list
    for utt_id, token_seq in lib_utt2token.items():
        assert padding_token not in token_seq
    print("padding token: ", padding_token)

    # print the total number of utterances in the librispeech dataset
    print("Total number of utterances in the librispeech dataset: ", len(lib_utt2token))

    # then iterate through all the challenge dataset, write the token sequence to the corresponding file
    # for those utterances that are not in the librispeech dataset, we pad them with <unk> tokens
    # make sure that the padded token sequence has the same length as the xls_r_1b token sequence

    #write the wavlm token file to the challenge dataset
    for dataset in challenge_dataset:
        wav2vec_token_file = os.path.join(wav2vec_challenge_dir, dataset, wav2vec_challenge_token)
        wavlm_token_file = os.path.join(wav2vec_challenge_dir, dataset, wavlm_lib_token)
        with open(wav2vec_token_file, "r", encoding="utf-8") as f:
            with open(wavlm_token_file, "w", encoding="utf-8") as f_w:
                for line in f:
                    utt_id, token_seq = line.strip().split(" ", 1)

                    # uttid = librispeech-103-1240-0000
                    # uttid or uttid removethe librispeech-
                    if utt_id in lib_utt2token :
                        f_w.write(utt_id + " " + lib_utt2token[utt_id] + "\n")
                    elif utt_id[12:] in lib_utt2token:
                        f_w.write(utt_id + " " + lib_utt2token[utt_id[12:]] + "\n")
                    else:
                        # pad the token sequence with <unk> tokens
                        # token_seq is a string, split it into a list of tokens
                        xls_r_1b_token_seq = list(token_seq)
                        pad_token_seq = [padding_token] * len(xls_r_1b_token_seq)
                        # without space between the tokens
                        f_w.write(utt_id + " " + "".join(pad_token_seq) + "\n")

    # copy the token list to the challenge dataset
    wavlm_token_list_file = os.path.join(wavlm_token_listdir, token_list_name, "src_tokens.txt")
    wav2cec_token_list_file = os.path.join(challenge_token_listdir, token_list_name )
    if not os.path.exists(wav2cec_token_list_file):
        os.makedirs(wav2cec_token_list_file)
    # print("cp " + wavlm_token_list_file + " " + wav2cec_token_list_file)
    # os.system("cp " + wavlm_token_list_file + " " + wav2cec_token_list_file+"/src_tokens.txt")

    # open the token list file, and read the tokens
    with open(wavlm_token_list_file, "r", encoding="utf-8") as f:
        tokens = f.readlines()
        # add the padding token to the tokens list if it is not in the list
        assert padding_token not in tokens
        # add the padding token to the second last position
        tokens.insert(-1, padding_token + "\n")
    # write the tokens to the challenge token list file
    with open(wav2cec_token_list_file+"/src_tokens.txt", "w", encoding="utf-8") as f:
        f.writelines(tokens)


if __name__ == "__main__":
    main()
