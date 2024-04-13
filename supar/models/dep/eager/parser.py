import os
from supar.models.dep.eager.oracle.node import Node

import torch
from supar.models.dep.eager.model import ArcEagerDependencyModel
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK, EOS
from supar.utils.field import Field, RawField, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger
from supar.utils.metric import AttachmentMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch
from supar.models.dep.eager.transform import ArcEagerTransform
from supar.models.dep.eager.oracle.arceager import ArcEagerDecoder

logger = get_logger(__name__)
from typing import Tuple, List, Union


class ArcEagerDependencyParser(Parser):
    MODEL = ArcEagerDependencyModel
    NAME = 'arceager-dependency'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FORM = self.transform.FORM
        self.STACK_TOP = self.transform.STACK_TOP
        self.BUFFER_FRONT = self.transform.BUFFER_FRONT
        self.TRANSITION, self.TREL = self.transform.TRANSITION, self.transform.TREL
        self.HEAD = self.transform.HEAD

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, texts, *feats, tags, _, _, stack_top, buffer_front, transitions, trels = batch
        tmask = self.get_padding_mask(stack_top)

        # pad stack_top and buffer_front vectors: note that padding index must be the length of the sequence
        pad_indices = (batch.lens - 1 - self.args.delay).tolist()
        stack_top, buffer_front = self.pad_tensor(stack_top, pad_indices), self.pad_tensor(buffer_front, pad_indices)

        # forward pass
        #   stack_top: torch.Tensor ~ [batch_size, pad(tr_len), n_transitions]
        #   buffer_front: torch.Tensor ~ [batch_size, pad(tr_len), n_trels]
        s_transition, s_trel, s_tag, qloss = self.model(words, stack_top, buffer_front, feats)

        # compute loss
        smask = batch.mask[:, (2+self.args.delay):]
        loss = self.model.loss(s_transition, s_trel, s_tag, transitions, trels, tags, smask, tmask, self.TRANSITION) + qloss
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> AttachmentMetric:
        words, texts, *feats, tags, heads, deprels, stack_top, buffer_front, transitions, trels = batch
        transition_mask = self.get_padding_mask(stack_top)


        # obtain transition loss
        stack_top, buffer_front = \
            self.pad_tensor(stack_top, (batch.lens - 1 - self.args.delay).tolist()), \
                self.pad_tensor(buffer_front, (batch.lens - 1 - self.args.delay).tolist())
        s_transition, s_trel, s_tag, qloss = self.model(words, stack_top, buffer_front, feats.copy())
        smask = batch.mask[:, (2+self.args.delay):]
        loss = self.model.loss(s_transition, s_trel, s_tag, transitions, trels, tags, smask, transition_mask, self.TRANSITION) + qloss

        # obtain indices of deprels from TREL field
        batch_size = words.shape[0]
        deprels = [self.TREL.vocab[deprels[b]] for b in range(batch_size)]

        # create decoders
        lens = list(map(len, texts))
        sentences = list()
        for b in range(batch_size):
            sentences.append(
                [
                    Node(ID=i + 1, FORM=form, UPOS='', HEAD=head, DEPREL=deprel) for i, (form, head, deprel) in \
                    enumerate(zip(texts[b], heads[b, :lens[b]].tolist(), deprels[b]))
                ]
            )
        decoders = list(map(
            lambda sentence: ArcEagerDecoder(sentence=sentence, bos='', eos='', unk=self.transform.TREL.unk_index),
            sentences
        ))

        # compute oracle simulation for all elements in batch
        head_preds, deprel_preds = self.oracle_decoding(decoders, words, feats)
        head_preds, deprel_preds = self.pad_tensor(head_preds), self.pad_tensor(deprel_preds)
        deprels = self.pad_tensor(deprels)

        seq_mask = batch.mask[:, (2 + self.args.delay):]
        return AttachmentMetric(loss, (head_preds, deprel_preds), (heads, deprels), seq_mask)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, texts, *feats = batch
        lens = (batch.lens - 2 - self.args.delay).tolist()
        batch_size = words.shape[0]

        # create decoders
        sentences = list()
        for b in range(batch_size):
            sentences.append(
                [
                    Node(ID=i + 1, FORM='', UPOS='', HEAD=None, DEPREL=None) for i in range(lens[b])
                ]
            )
        decoders = list(map(
            lambda sentence: ArcEagerDecoder(sentence=sentence, bos='', eos='', unk=self.transform.TREL.unk_index),
            sentences
        ))

        # compute oracle simulation for all elements in batch
        head_preds, deprel_preds = self.oracle_decoding(decoders, words, feats)

        batch.heads = head_preds
        batch.deprels = [self.TREL.vocab[i] for i in deprel_preds]

        return batch

    def get_padding_mask(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """
        From a list of tensors of different lengths, creates a padding mask where False values indicates
        padding tokens. True otherwise.
        Args:
            tensor_list: List of tensors.
        Returns: torch.Tensor ~ [len(tensor_list), max(lenghts)]

        """
        lens = list(map(len, tensor_list))
        max_len = max(lens)
        return torch.tensor([[True] * length + [False] * (max_len - length) for length in lens]).to(self.model.device)

    def pad_tensor(
            self,
            tensor_list: Union[List[torch.Tensor], List[List[int]]],
            pad_index: Union[int, List[int]] = 0
    ):
        """
        Applies padding to a list of tensors or list of lists.
        Args:
            tensor_list: List of tensors or list of lists.
            pad_index: Index used for padding or list of indices used for padding for each item of tensor_list.
        Returns: torch.Tensor ~ [len(tensor_list), max(lengths)]
        """
        max_length = max(map(len, tensor_list))
        if isinstance(pad_index, int):
            if isinstance(tensor_list[0], list):
                return torch.tensor(
                    [tensor + [pad_index] * (max_length - len(tensor)) for tensor in tensor_list]).to(self.model.device)
            else:
                return torch.tensor(
                    [tensor.tolist() + [pad_index] * (max_length - len(tensor)) for tensor in tensor_list]).to(self.model.device)
        else:
            pad_indexes = pad_index
            if isinstance(tensor_list[0], list):
                return torch.tensor(
                    [tensor + [pad_index] * (max_length - len(tensor)) for tensor, pad_index in
                     zip(tensor_list, pad_indexes)]).to(self.model.device)
            else:
                return torch.tensor(
                    [tensor.tolist() + [pad_index] * (max_length - len(tensor)) for tensor, pad_index in
                     zip(tensor_list, pad_indexes)]).to(self.model.device)

    def get_text_mask(self, batch):
        text_lens = (batch.lens - 2 - self.args.delay).tolist()
        mask = batch.mask
        mask[:, 0] = 0  # remove bos token
        for i, text_len in enumerate(text_lens):
            mask[i, (1 + text_len):] = 0
        return mask

    def oracle_decoding(self, decoders: List[ArcEagerDecoder], words: torch.Tensor, feats: List[torch.Tensor]) -> Tuple[
        List[List[int]]]:
        """
        Implements Arc-Eager decoding. Using words indices, creates the initial state of the Arc-Eager oracle
        and predicts each (transition, trel) with the TransitionDependencyModel.
        Args:
            decoders: List[ArcEagerDecoder] ~ batch_size
            words: torch.Tensor ~ [batch_size, seq_len]
            feats: List[torch.Tensor ~ [batch_size, seq_len, feat_embed]] ~ n_feat


        Returns: head_preds, deprel_preds
            head_preds: List[List[int] ~ sen_len] ~ batch_size: Head values for each sentence in batch.
            deprel_preds: List[List[int] ~ sen_len] ~ batch_size: Indices of dependency relations for each sentence in batch.
        """
        # create a mask vector to filter those decoders that achieved the final state
        compute = [True for _ in range(len(decoders))]
        batch_size = len(decoders)
        exclude = self.TREL.vocab[['<reduce>', '<shift>']]

        # obtain word representations of the encoder
        x, *_ = self.model.encoder_forward(words, feats)

        stack_top = [torch.tensor([decoders[b].stack.get().ID]).reshape(1) for b in range(batch_size)]
        buffer_front = [torch.tensor([decoders[b].buffer.get().ID]).reshape(1) for b in range(batch_size)]
        counter = 0
        while any(compute):
            s_transition, s_trel = self.model.decoder_forward(x, torch.stack(stack_top), torch.stack(buffer_front))
            transition_preds, trel_preds = self.model.decode(s_transition, s_trel, exclude)
            transition_preds, trel_preds = transition_preds[:, counter, :], trel_preds[:, counter]
            transition_preds = [self.TRANSITION.vocab[i.tolist()] for i in
                                transition_preds.reshape(batch_size, self.args.n_transitions)]

            for b, decoder in enumerate(decoders):
                if not compute[b]:
                    stop, bfront = stack_top[b][-1].item(), buffer_front[b][-1].item()
                else:
                    result = decoder.apply_transition(transition_preds[b], trel_preds[b].item())
                    if result is None:
                        stop, bfront = stack_top[b][-1].item(), buffer_front[b][-1].item()
                        compute[b] = False
                    else:
                        stop, bfront = result[0].ID, result[1].ID
                stack_top[b] = torch.concat([stack_top[b], torch.tensor([stop])])
                buffer_front[b] = torch.concat([buffer_front[b], torch.tensor([bfront])])
            counter += 1

        head_preds = [[node.HEAD for node in decoder.decoded_nodes] for decoder in decoders]
        deprel_preds = [[node.DEPREL for node in decoder.decoded_nodes] for decoder in decoders]
        return head_preds, deprel_preds

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

        # ------------------------------- source fields -------------------------------
        WORD, TAG, CHAR = None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            pad_token = t.pad if t.pad else PAD
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, fix_len=args.fix_len, tokenize=t, delay=args.delay)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, delay=args.delay)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, eos=EOS, fix_len=args.fix_len, delay=args.delay)
        TAG = Field('tags')
        TEXT = RawField('texts')
        STACK_TOP = RawField('stack_top', fn=lambda x: torch.tensor(x))
        BUFFER_FRONT = RawField('buffer_front', fn=lambda x: torch.tensor(x))

        # ------------------------------- target fields -------------------------------
        TRANSITION = Field('transition')
        TREL = Field('trels')
        HEAD = Field('heads', use_vocab=False)
        DEPREL = RawField('rels')

        transform = ArcEagerTransform(
            FORM=(WORD, TEXT, CHAR), UPOS=TAG, HEAD=HEAD, DEPREL=DEPREL,
            STACK_TOP=STACK_TOP, BUFFER_FRONT=BUFFER_FRONT, TRANSITION=TRANSITION, TREL=TREL,
        )

        train = Dataset(transform, args.train, **args)

        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None),
                       lambda x: x / torch.std(x))
            if TAG:
                TAG.build(train)
            if CHAR:
                CHAR.build(train)
        TAG.build(train)
        TREL.build(train)
        TRANSITION.build(train)

        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_transitions': len(TRANSITION.vocab),
            'n_trels': len(TREL.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })

        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
