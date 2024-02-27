#!/usr/bin/env python3
from espnet2.tasks.mt_multi_input import MTTask_multi_input


def get_parser():
    parser = MTTask_multi_input.get_parser()
    return parser


def main(cmd=None):
    r"""MT training.

    Example:

        % python mt_train.py st --print_config --optim adadelta \
                > conf/train_mt.yaml
        % python mt_train.py --config conf/train_mt.yaml
    """
    MTTask_multi_input.main(cmd=cmd)


if __name__ == "__main__":
    main()
