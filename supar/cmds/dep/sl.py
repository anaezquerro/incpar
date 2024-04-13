# -*- coding: utf-8 -*-

import argparse

from supar import SLDependencyParser
from supar.cmds.run import init


def main():
    parser = argparse.ArgumentParser(description='Create SL Dependency Parsing Parser.')
    parser.set_defaults(Parser=SLDependencyParser)
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--implicit', action='store_true', help='whether to conduct implicit binarization')
    subparser.add_argument('--max_len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.pid', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.pid', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.pid', help='path to test file')
    subparser.add_argument('--embed', default=None, help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--use_vq', action='store_true', default=False, help='whether to use vector quantization')
    subparser.add_argument('--decoder', choices=['mlp', 'lstm'], default='mlp', help='incremental decoder to use')
    subparser.add_argument('--sl_codes', choices=['abs', 'rel', 'pos', '1p', '2p'], default=None, help='SL coding used')
    subparser.add_argument('--delay', type=int, default=0)
    subparser.add_argument('--root_node', type=str, default='S')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.pid', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.pid', help='path to dataset')
    subparser.add_argument('--pred', default='pred.pid', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()
