# -*- coding: utf-8 -*-

import os
from typing import Dict, Iterable, Set, Union

import torch, nltk
from supar.models.const.sl.model import SLConstituentModel
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import SpanMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.models.const.sl.transform import SLConstituent
from supar.utils.transform import Batch
from supar.models.const.crf.transform import Tree
from supar.codelin import get_con_encoder, LinearizedTree, C_Label
from supar.codelin.utils.constants import BOS as C_BOS, EOS as C_EOS

logger = get_logger(__name__)


class SLConstituentParser(Parser):

    NAME = 'SLConstituentParser'
    MODEL = SLConstituentModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.COMMON = self.transform.COMMON
        self.ANCESTOR = self.transform.ANCESTOR
        self.encoder = self.transform.encoder

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().train(**Config().update(locals()))

    def evaluate(
        self,
        data: Union[str, Iterable],
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().evaluate(**Config().update(locals()))

    def predict(
        self,
        data: Union[str, Iterable],
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, texts, *feats, commons, ancestors, trees = batch
        mask = batch.mask[:, (1+self.args.delay):]
        s_common, s_ancestor, qloss = self.model(words, feats)
        loss = self.model.loss(s_common, s_ancestor, commons, ancestors, mask) + qloss
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanMetric:
        words, texts, *feats, commons, ancestors, trees = batch
        mask = batch.mask[:, (1+self.args.delay):]

        # forward pass
        s_common, s_ancestor, qloss = self.model(words, feats)
        loss = self.model.loss(s_common, s_ancestor, commons, ancestors, mask) + qloss

        # make predictions of the output of the network
        common_preds, ancestor_preds = self.model.decode(s_common, s_ancestor)

        # obtain original tokens to compute decoding
        lens = (batch.lens - 1 - self.args.delay).tolist()
        common_preds = [self.COMMON.vocab[i.tolist()] for i in common_preds[mask].split(lens)]
        ancestor_preds = [self.ANCESTOR.vocab[i.tolist()] for i in ancestor_preds[mask].split(lens)]
        # tag_preds = [self.transform.POS.vocab[i.tolist()] for i in tags[mask].split(lens)]
        tag_preds = map(lambda tree: tuple(zip(*tree.pos()))[1], trees)

        preds = list()
        for i, (forms, upos, common_pred, ancestor_pred) in enumerate(zip(texts, tag_preds, common_preds, ancestor_preds)):
            labels = list(map(lambda x: self.encoder.separator.join(x), zip(common_pred, ancestor_pred)))
            linearized = list(map(lambda x: '\t'.join(x), zip(forms, upos, labels)))
            linearized = [f'{C_BOS}\t{C_BOS}\t{C_BOS}'] + linearized + [f'{C_EOS}\t{C_EOS}\t{C_EOS}']
            linearized = '\n'.join(linearized)
            tree = LinearizedTree.from_string(linearized, mode='CONST', separator=self.encoder.separator,
                                              unary_joiner=self.encoder.unary_joiner)
            tree = self.encoder.decode(tree)
            tree = tree.postprocess_tree('strat_max', clean_nulls=False)
            preds.append(nltk.Tree.fromstring(str(tree)))

            if len(preds[-1].leaves()) != len(trees[i].leaves()):
                with open('error', 'w') as file:
                    file.write(linearized)


        return SpanMetric(loss,
                          [Tree.factorize(tree, None, None) for tree in preds],
                          [Tree.factorize(tree, None, None) for tree in trees])

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, texts, *feats = batch
        tags = feats[-1][:, (1+self.args.delay):]
        mask = batch.mask[:, (1 + self.args.delay):]

        # forward pass
        s_common, s_ancestor, _ = self.model(words, feats)

        # make predictions of the output of the network
        common_preds, ancestor_preds = self.model.decode(s_common, s_ancestor)

        # obtain original tokens to compute decoding
        lens = (batch.lens - 1 - self.args.delay).tolist()
        common_preds = [self.COMMON.vocab[i.tolist()] for i in common_preds[mask].split(lens)]
        ancestor_preds = [self.ANCESTOR.vocab[i.tolist()] for i in ancestor_preds[mask].split(lens)]
        tag_preds = [self.transform.POS.vocab[i.tolist()] for i in tags[mask].split(lens)]

        preds = list()
        for i, (forms, upos, common_pred, ancestor_pred) in enumerate(zip(texts, tag_preds, common_preds, ancestor_preds)):
            labels = list(map(lambda x: self.encoder.separator.join(x), zip(common_pred, ancestor_pred)))
            linearized = list(map(lambda x: '\t'.join(x), zip(forms, upos, labels)))
            linearized = [f'{C_BOS}\t{C_BOS}\t{C_BOS}'] + linearized + [f'{C_EOS}\t{C_EOS}\t{C_EOS}']
            linearized = '\n'.join(linearized)
            tree = LinearizedTree.from_string(linearized, mode='CONST', separator=self.encoder.separator, unary_joiner=self.encoder.unary_joiner)
            tree = self.encoder.decode(tree)
            tree = tree.postprocess_tree('strat_max', clean_nulls=False)
            preds.append(nltk.Tree.fromstring(str(tree)))
        batch.trees = preds
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.WORD[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        TAG, CHAR = None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            pad_token = t.pad if t.pad else PAD
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t, delay=args.delay)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True, delay=args.delay)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len, delay=args.delay)
        TAG = Field('tags', bos=BOS,)
        TEXT = RawField('texts')
        COMMON = Field('commons')
        ANCESTOR = Field('ancestors')
        TREE = RawField('trees')
        encoder = get_con_encoder(args.codes)
        transform = SLConstituent(encoder=encoder, WORD=(WORD, TEXT, CHAR), POS=TAG,
                                  COMMON=COMMON, ANCESTOR=ANCESTOR, TREE=TREE)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if CHAR is not None:
                CHAR.build(train)
        TAG.build(train)
        COMMON.build(train)
        ANCESTOR.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_commons': len(COMMON.vocab),
            'n_ancestors': len(ANCESTOR.vocab),
            'n_tags': len(TAG.vocab),
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'delay': 0 if 'delay' not in args.keys() else args.delay,
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
