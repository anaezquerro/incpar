# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.model import Model
from supar.modules import MLP, DecoderLSTM
from supar.utils import Config
from typing import List


class SLConstituentModel(Model):

    def __init__(self,
                 n_words,
                 n_commons,
                 n_ancestors,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat: List[str] =['char'],
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
        self.common_decoder, self.ancestor_decoder = None, None
        if self.args.decoder == 'lstm':
            decoder = lambda out_dim: DecoderLSTM(
                self.args.n_encoder_hidden, self.args.n_encoder_hidden, out_dim, 
                self.args.n_decoder_layers, dropout=mlp_dropout, device=self.device
            )
        else:
            decoder = lambda out_dim: MLP(
                n_in=self.args.n_encoder_hidden, n_out=out_dim,
                dropout=mlp_dropout, activation=True
            )

        self.common_decoder = decoder(self.args.n_commons)
        self.ancestor_decoder = decoder(self.args.n_ancestors)

        # create delay projection
        if self.args.delay != 0:
            self.delay_proj = MLP(n_in=self.args.n_encoder_hidden * (self.args.delay + 1),
                                  n_out=self.args.n_encoder_hidden, dropout=mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: torch.Tensor, feats: List[torch.Tensor]=None):
        # words, *feats ~ [batch_size, bos + pad(seq_len) + delay, n_encoder_hidden]
        x = self.encode(words, feats)
        x = x[:, 1:, :]

        # x ~ [batch_size, pad(seq_len), n_encoder_hidden]
        batch_size, pad_seq_len, embed_size = x.shape
        if self.args.delay != 0:
            x = torch.cat([x[:, i:(pad_seq_len - self.args.delay + i), :] for i in range(self.args.delay+1)], dim=2)
            x = self.delay_proj(x)

        x, qloss = self.vq_forward(x)   # vector quantization

        # s_common ~ [batch_size, pad(seq_len), n_commons]
        # s_ancestor ~ [batch_size, pad(seq_len), n_ancestors]
        s_common, s_ancestor = self.common_decoder(x), self.ancestor_decoder(x)
        return s_common, s_ancestor, qloss

    def loss(self,
             s_common: torch.Tensor, s_ancestor: torch.Tensor, 
             commons: torch.Tensor, ancestors: torch.Tensor, mask: torch.Tensor):
        s_common, commons = s_common[mask], commons[mask]
        s_ancestor, ancestors = s_ancestor[mask], ancestors[mask]
        common_loss = self.criterion(s_common, commons)
        ancestor_loss = self.criterion(s_ancestor, ancestors)
        return common_loss + ancestor_loss

    def decode(self, s_common, s_ancestor):
        common_pred = s_common.argmax(-1)
        ancestor_pred = s_ancestor.argmax(-1)
        return common_pred, ancestor_pred