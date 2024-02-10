#!/usr/bin/env python

# Copyright    2017 Hossein Hadian
# Apache 2.0


""" This reads data/train/text from standard input, converts the word transcriptions
    to phone transcriptions using the provided lexicon,
    and writes them to standard output.
"""
from __future__ import print_function

import argparse
from os.path import join
import sys
import copy
import random

parser = argparse.ArgumentParser(description="""This script reads
    data/train/text from std input and converts the word transcriptions
    to phone transcriptions using the provided lexicon""")
parser.add_argument('--langdir', type=str, default='/export/fs05/mliu121/kaldi/egs/librispeech/s5/data/lang_nosp')
parser.add_argument('--edge-silprob', type=float, default=0.8,
                    help="""Probability of optional silence at the beginning
                    and end.""")
parser.add_argument('--between-silprob', type=float, default=0.2,
                    help="Probability of optional silence between the words.")
parser.add_argument('--text_file', type=str, default='text',
                    help="""Input file to read from, default is data/train/text""")
parser.add_argument('--phone_file', type=str, default='text.phones',
                    help="""Output file to write to, default is data/train/text.phones""")
args = parser.parse_args()





# optional silence
sil = open(join(args.langdir,
                "phones/optional_silence.txt")).readline().strip()

oov_word = open(join(args.langdir, "oov.txt")).readline().strip()


# load the lexicon
lexicon = {}
with open(join(args.langdir, "phones/align_lexicon.txt")) as f:
    for line in f:
        line = line.strip();
        parts = line.split()
        lexicon[parts[0]] = parts[2:]  # ignore parts[1]

n_tot = 0
n_fail = 0
# read the input file
with open(args.text_file, 'r') as f:
    lines = f.readlines()

# write the output file
with open(args.phone_file, 'w') as f:
    # process each line of the input file
    for line in lines:
        line = line.strip().split()
        key = line[-1]  # the last element is the utterance id
        word_trans = line[:-1]  # word-level transcription
        phone_trans = []  # phone-level transcription
        for i in range(len(word_trans)):
            n_tot += 1
            word = word_trans[i]
            if word not in lexicon:
                n_fail += 1
                if n_fail < 20:
                    sys.stderr.write("{} not found in lexicon, replacing with {}\n".format(word, oov_word))
                elif n_fail == 20:
                    sys.stderr.write("Not warning about OOVs any more.\n")
                pronunciation = lexicon[oov_word]
            else:
                pronunciation = copy.deepcopy(lexicon[word])
            phone_trans += pronunciation
        # write the phone-level transcription and the utterance id to the output file
        phone_line = " ".join(phone_trans) + " " + key + "\n"
        f.write(phone_line)

sys.stderr.write("Done. {} out of {} were OOVs.\n".format(n_fail, n_tot))



