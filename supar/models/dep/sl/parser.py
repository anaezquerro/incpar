# -*- coding: utf-8 -*-

import os, re
from typing import Iterable, Union, List

import torch
from supar.models.dep.sl.model import SLDependencyModel
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch
from supar.models.dep.sl.transform import SLDependency
from supar.codelin import get_dep_encoder, LinearizedTree, D_Label
from supar.codelin.utils.constants import D_ROOT_HEAD
from supar.codelin import LABEL_SEPARATOR
logger = get_logger(__name__)
from typing import Tuple, List, Union


NONE = '<none>'
OPTIONS = [r'>\*', r'<\*', r'/\*', r'\\\*']


def split_planes(labels: Tuple[str], plane: int) -> Tuple[str]:

    split = lambda label: min(match.span()[0] if match is not None else len(label) for match in map(lambda x: re.search(x, label), OPTIONS))
    splits = list(map(split, labels))
    if plane == 0:
        return tuple(label[:i] for label, i in zip(labels, splits))
    else:
        return tuple(label[i:] if i < len(label) else NONE for label, i in zip(labels, splits))

class SLDependencyParser(Parser):

    NAME = 'SLDependencyParser'
    MODEL = SLDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.TAG = self.transform.UPOS
        self.LABEL, self.DEPREL = self.transform.LABEL, self.transform.DEPREL
        self.encoder = get_dep_encoder(self.args.codes, LABEL_SEPARATOR)

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 1000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        punct: bool = False,
        tree: bool = False,
        proj: bool = False,
        partial: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().train(**Config().update(locals()))

    def few_shot(
            self,
            train: str,
            dev: Union[str, Iterable],
            test: Union[str, Iterable],
            n_samples: int,
            epochs: int = 2,
            batch_size: int = 50
    ) -> None:
        return super().few_shot(**Config().update(locals()))

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
            tree: bool = True,
            proj: bool = False,
            verbose: bool = True,
            **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def evaluate(
        self,
        data: Union[str, Iterable],
        batch_size: int = 500,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        punct: bool = False,
        tree: bool = True,
        proj: bool = False,
        partial: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().evaluate(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        if self.args.encoder == 'lstm':
            words, texts, chars, tags, _, rels, *labels = batch
            feats = [chars]
        else:
            words, texts, tags, _, rels, *labels = batch
            feats = []
        labels = labels[0] if len(labels) == 1 else labels
        mask = batch.mask[:, (1+self.args.delay):]

        # forward pass
        s_label, s_rel, s_tag, qloss = self.model(words, feats)

        # compute loss
        loss = self.model.loss(s_label, s_rel, s_tag, labels, rels, tags, mask) + qloss

        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        if self.args.encoder == 'lstm':
            words, texts, chars, tags, heads, rels, *labels = batch
            feats = [chars]
        else:
            words, texts, tags, heads, rels, *labels = batch
            feats = []
        labels = labels[0] if len(labels) == 1 else labels
        mask = batch.mask[:, (1+self.args.delay):]
        lens = (batch.lens - 1 - self.args.delay).tolist()

        # forward pass
        s_label, s_rel, s_tag, qloss = self.model(words, feats)

        # compute loss and decode
        loss = self.model.loss(s_label, s_rel, s_tag, labels, rels, tags, mask) + qloss
        label_preds, rel_preds, tag_preds = self.model.decode(s_label, s_rel, s_tag, mask)

        # obtain original label and deprel strings to compute decoding
        if self.args.codes == '2p':
            label_preds = [
                (self.LABEL[0].vocab[l1.tolist()], self.LABEL[1].vocab[l2.tolist()])
                for l1, l2 in zip(*map(lambda x: x[mask].split(lens), label_preds))
            ]
            label_preds = [
                list(map(lambda x: x[0] + (x[1] if x[1] != NONE else ''), zip(*label_pred)))
                for label_pred in label_preds
            ]
        else:
            label_preds = [self.LABEL.vocab[i.tolist()] for i in label_preds[mask].split(lens)]
        deprel_preds = [self.DEPREL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]

        if self.args.encoder == 'lstm':
            tag_preds = [self.TAG.vocab[i.tolist()] for i in tag_preds[mask].split(lens)]
        else:
            tag_preds = [self.TAG.vocab[i.tolist()] for i in tags[mask].split(lens)]

        # decode
        head_preds = list()
        for label_pred, deprel_pred, tag_pred, forms in zip(label_preds, deprel_preds, tag_preds, texts):
            labels = [D_Label(label, deprel, self.encoder.separator) for label, deprel in zip(label_pred, deprel_pred)]
            linearized_tree = LinearizedTree(list(forms), tag_pred, ['_']*len(forms), labels, 0)

            decoded_tree = self.encoder.decode(linearized_tree)
            decoded_tree.postprocess_tree(D_ROOT_HEAD)
            head_preds.append(torch.tensor([int(node.head) for node in decoded_tree.nodes]))


        # resize head predictions (add padding)
        resize = lambda list_of_tensors: \
            torch.stack([
                torch.concat([x, torch.zeros(mask.shape[1] - len(x))])
                for x in list_of_tensors])

        head_preds = resize(head_preds).to(torch.int32).to(self.model.device)

        return AttachmentMetric(loss, (head_preds, rel_preds), (heads, rels), mask)


    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        if self.args.encoder == 'lstm':
            words, texts, *feats, tags = batch
        else:
            words, texts, *feats, tags = batch
        mask = batch.mask[:, (1 + self.args.delay):]
        lens = (batch.lens - 1 - self.args.delay).tolist()

        # forward pass
        s_label, s_rel, s_tag, qloss = self.model(words, feats)

        # compute loss and decode
        label_preds, rel_preds, tag_preds = self.model.decode(s_label, s_rel, s_tag, mask)

        # obtain original label and deprel strings to compute decoding
        if self.args.codes == '2p':
            label_preds = [
                (self.LABEL[0].vocab[l1.tolist()], self.LABEL[1].vocab[l2.tolist()])
                for l1, l2 in zip(*map(lambda x: x[mask].split(lens), label_preds))
            ]
            label_preds = [
                list(map(lambda x: x[0] + (x[1] if x[1] != NONE else ''), zip(*label_pred)))
                for label_pred in label_preds
            ]
        else:
            label_preds = [self.LABEL.vocab[i.tolist()] for i in label_preds[mask].split(lens)]

        deprel_preds = [self.DEPREL.vocab[i.tolist()] for i in rel_preds[mask].split(lens)]

        if self.args.encoder == 'lstm':
            tag_preds = [self.TAG.vocab[i.tolist()] for i in tag_preds[mask].split(lens)]
        else:
            tag_preds = [self.TAG.vocab[i.tolist()] for i in tags[mask].split(lens)]

        # decode
        head_preds = list()
        for label_pred, deprel_pred, tag_pred, forms in zip(label_preds, deprel_preds, tag_preds, texts):
            labels = [D_Label(label, deprel, self.encoder.separator) for label, deprel in zip(label_pred, deprel_pred)]
            linearized_tree = LinearizedTree(forms, tag_pred, ['_'] * len(forms), labels, 0)

            decoded_tree = self.encoder.decode(linearized_tree)
            decoded_tree.postprocess_tree(D_ROOT_HEAD)

            head_preds.append([int(node.head) for node in decoded_tree.nodes])

        batch.heads = head_preds
        batch.rels = deprel_preds

        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        args = Config(**locals())

        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.FORM[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        CHAR = None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            pad_token = t.pad if t.pad else PAD
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t, delay=args.delay)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True, delay=args.delay)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len, delay=args.delay)
        TEXT = RawField('texts')
        TAG = Field('tags')


        if args.codes == '2p':
            LABEL1 = Field('labels1', fn=lambda seq: split_planes(seq, 0))
            LABEL2 = Field('labels2', fn=lambda seq: split_planes(seq, 1))
            LABEL = (LABEL1, LABEL2)
        else:
            LABEL = Field('labels')

        DEPREL = Field('rels')
        HEAD = Field('heads', use_vocab=False)

        transform = SLDependency(
            encoder=get_dep_encoder(args.codes, LABEL_SEPARATOR),
            FORM=(WORD, TEXT, CHAR), UPOS=TAG, HEAD=HEAD, DEPREL=DEPREL, LABEL=LABEL)

        train = Dataset(transform, args.train, **args)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if CHAR:
                CHAR.build(train)
        TAG.build(train)
        DEPREL.build(train)

        if args.codes == '2p':
            LABEL[0].build(train)
            LABEL[1].build(train)
        else:
            LABEL.build(train)

        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels': len(LABEL.vocab) if args.codes != '2p' else (len(LABEL[0].vocab), len(LABEL[1].vocab)),
            'n_rels': len(DEPREL.vocab),
            'n_tags': len(TAG.vocab),
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'delay': args.delay
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser



