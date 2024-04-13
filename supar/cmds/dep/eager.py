# -*- coding: utf-8 -*-

import argparse

from supar import ArcEagerDependencyParser
from supar.cmds.run import init

def main():
    parser = argparse.ArgumentParser(description='Create Transition Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=ArcEagerDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', default=None, help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--use_vq', action='store_true', default=False, help='whether to use vector quantization')
    subparser.add_argument('--decoder', choices=['mlp', 'lstm'], default='mlp', help='incremental decoder to use')
    subparser.add_argument('--delay', type=int, default=0)
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()
