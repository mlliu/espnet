#!/usr/bin/env bash
# this script is used to combine the dump/raw of librispeech_100 and ml-superb

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
data_url=www.openslr.org/resources/12
train_set="train"
train_dev="dev"
dumpdir="/export/fs05/mliu121/espnet_data/challenge/dump/audio_raw"


log "$0 $*"
. ./utils/parse_options.sh || exit 1;

. ./db.sh
. ./path.sh
. ./cmd.sh

    log "Combine the training and valid sets"
    mkdir -p $dumpdir/${train_set}
    mkdir -p $dumpdir/${train_dev}
    #rm $dumpdir/${train_set}/* $dumpdir/${train_dev}/*

    # librispeech_100
    prefix="librispeech-"
    for dset in "train_clean_100" "dev"; do
        if [ ${dset} = "train_clean_100" ]; then
            _dir="$dumpdir/${train_set}"
        else
            _dir="$dumpdir/${train_dev}"
        fi

        src_dir="$dumpdir/librispeech_100/${dset}"
        <${src_dir}/utt2spk awk -v prefix="${prefix}" '{print(prefix $1, prefix $2)}' >> ${_dir}/utt2spk
        for f in text wav.scp utt2num_samples; do
            <${src_dir}/${f} awk -v prefix="${prefix}" '{print(prefix $0)}' >> ${_dir}/${f}
        done
    done

    # ml-superb
    prefix="ml_suprb-"
    for dset in "train_1h" "dev_1h"; do
        if [ ${dset} = "train_1h" ]; then
            _dir="$dumpdir/${train_set}"
        else
            _dir="$dumpdir/${train_dev}"
        fi

        src_dir="$dumpdir/ml_superb/${dset}"
        <${src_dir}/utt2spk awk -v prefix="${prefix}" '{print(prefix $1, prefix $2)}' >> ${_dir}/utt2spk
        for f in text wav.scp utt2num_samples; do
            <${src_dir}/${f} awk -v prefix="${prefix}" '{print(prefix $0)}' >> ${_dir}/${f}
        done
    done

    for dset in train dev; do
        utils/utt2spk_to_spk2utt.pl $dumpdir/${dset}/utt2spk > $dumpdir/${dset}/spk2utt
        utils/fix_data_dir.sh $dumpdir/${dset}
    done
    # feat_type =raw
