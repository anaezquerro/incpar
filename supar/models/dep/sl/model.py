# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.model import Model
from supar.modules import MLP, DecoderLSTM
from supar.utils import Config
from typing import Tuple, List, Union

class SLDependencyModel(Model):
    def __init__(self,
                 n_words: int,
                 n_labels: Union[Tuple[int], int],
                 n_rels: int,
                 n_tags: int = None,
                 n_chars: int = None,
                 encoder: str ='lstm',
                 feat: List[str] = [],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_encoder_hidden=800,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # create decoder
        self.label_decoder, self.rel_decoder = None, None
        if self.args.decoder == 'lstm':
            decoder = lambda out_dim: DecoderLSTM(
                input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_encoder_hidden,
                num_layers=self.args.n_decoder_layers, dropout=mlp_dropout,
                output_size=out_dim)
        else:
            decoder = lambda out_dim: MLP(
                n_in=self.args.n_encoder_hidden, n_out=out_dim, dropout=mlp_dropout
            )

        if self.args.codes == '2p':
            self.label_decoder1 = decoder(self.args.n_labels[0])
            self.label_decoder2 = decoder(self.args.n_labels[1])
            self.label_decoder = lambda x: (self.label_decoder1(x), self.label_decoder2(x))
        else:
            self.label_decoder = decoder(self.args.n_labels)

        self.rel_decoder = decoder(self.args.n_rels)

        # create delay projection
        if self.args.delay != 0:
            self.delay_proj = MLP(n_in=self.args.n_encoder_hidden * (self.args.delay + 1),
                                  n_out=self.args.n_encoder_hidden, dropout=mlp_dropout)

        # create PoS tagger
        if self.args.encoder == 'lstm':
            self.pos_tagger = DecoderLSTM(
                input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_encoder_hidden,
                output_size=self.args.n_tags, num_layers=1, dropout=mlp_dropout, device=self.device)
        else:
            self.pos_tagger = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: torch.Tensor, feats: List[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            s_label (~Union[torch.Tensor, Tuple[torch.Tensor]]): ``[batch_size, seq_len, n_labels]``
                Tensor or 2-dimensional tensor tuple (if 2-planar bracketing coding is being used) which holds the
                scores of all possible labels.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, n_rels]``
                Holds scores of all possible dependency relations.
            s_tag (~torch.Tensor): ` [batch_size, seq_len, n_tags]``
                Holds scores of all possible tags for each word.
            qloss (~torch.Tensor):
                Vector quantization loss.
        """

        # x ~ [batch_size, bos + pad(seq_len) + delay, n_encoder_hidden]
        x = self.encode(words, feats)
        x = x[:, 1:, :]                 # remove BoS token

        # s_tag ~ [batch_size, pad(seq_len), n_tags]
        s_tag = self.pos_tagger(x if self.args.delay == 0 else x[:, :-self.args.delay, :])

        # map or concatenate delayed representations
        if self.args.delay != 0:
            x = torch.cat([x[:, i:(x.shape[1] - self.args.delay + i), :] for i in range(self.args.delay+1)], dim=2)
            x = self.delay_proj(x)

        # x ~ [batch_size, pad(seq_len), n_encoder_hidden]
        batch_size, pad_seq_len, _ = x.shape

        # pass through vector quantization module
        x, qloss = self.vq_forward(x)

        # make predictions of labels/relations
        s_label = self.label_decoder(x)
        s_rel = self.rel_decoder(x)

        return s_label, s_rel, s_tag, qloss

    def loss(
        self,
        s_label: Union[Tuple[torch.Tensor], torch.Tensor],
        s_rel: torch.Tensor,
        s_tag: torch.Tensor,
        labels: Union[Tuple[torch.Tensor], torch.Tensor],
        rels: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
         ) -> torch.Tensor:

        loss = self.criterion(s_label[mask], labels[mask]) if self.args.codes != '2p' else sum(self.criterion(scores[mask], golds[mask]) for scores, golds in zip(s_label, labels))

        loss += self.criterion(s_rel[mask], rels[mask])

        if self.args.encoder == 'lstm':
            loss += self.criterion(s_tag[mask], tags[mask])
        return loss

    def decode(self, s_label: Union[Tuple[torch.Tensor], torch.Tensor], s_rel: torch.Tensor, s_tag: torch.Tensor,
               mask: torch.Tensor):
        label_preds = s_label.argmax(-1) if self.args.codes != '2p' else tuple(map(lambda x: x.argmax(-1), s_label))
        rel_preds = s_rel.argmax(-1)
        tag_preds = s_tag.argmax(-1) if self.args.encoder == 'lstm' else None
        return label_preds, rel_preds, tag_preds
