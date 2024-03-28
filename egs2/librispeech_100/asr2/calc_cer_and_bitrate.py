# in this script, we calculate the CER and bitrate of the model
# the input file is the result file of the decoding, which can be used to calcaute the CER
# anothe input file is the feature file, which can be used to calculate the bitrate

import os
import numpy as np
import argparse
def calc_cer(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # find the line ### CER,
            # then in the following line, find the Wrds and CER for dev_clean, dev_other, test_clean, test_other
            if "### CER" in line:
                flag = 1
                continue
            if flag == 1 and "dev_clean" in line:
                dev_clean_wrds  = int(line.strip("|").split()[3])
                dev_clean_cer = float(line.strip("|").split()[8])
            if flag == 1 and "dev_other" in line:
                dev_other_wrds  = int(line.strip("|").split()[3])
                dev_other_cer = float(line.strip("|").split()[8])
            if flag == 1 and "test_clean" in line:
                test_clean_wrds  = int(line.strip("|").split()[3])
                test_clean_cer = float(line.strip("|").split()[8])
            if flag == 1 and "test_other" in line:
                test_other_wrds  = int(line.strip("|").split()[3])
                test_other_cer = float(line.strip("|").split()[8])

            if "### TER" in line:
                break
        # calculate the average CER
        total_wrds = dev_clean_wrds + dev_other_wrds + test_clean_wrds + test_other_wrds
        total_error = (dev_clean_cer * dev_clean_wrds + dev_other_cer * dev_other_wrds + test_clean_cer * test_clean_wrds + test_other_cer * test_other_wrds)/ 100
        average_cer = total_error / total_wrds
        print("The average CER is: ", average_cer)


# def cala_bitrate(feature_file, vocabulary_file):
#     # bit rate = length of feautre file * log2(vocab_size) / N
#     # N is the total length of the audio in seconds,





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, help="the result file of the decoding")
    parser.add_argument("feature", type=str, help="the feature file of the decoding")
    parser.add_argument("vocab", type=str, help="the feature file of the decoding")
    parser.add_argument("number_of_features", type=str, help="the feature file of the decoding")
    args = parser.parse_args()
    result_file = args.result
    feature_file = args.feature
    vocab = args.vocab
    calc_cer(result_file)
    # cala_bitrate(feature_file, vocab)
