import torch
import torch.nn as nn
from supar.model import Model
from supar.modules import MLP, DecoderLSTM
from supar.utils import Config
from typing import Tuple, List

class ArcEagerDependencyModel(Model):

    def __init__(self,
                 n_words,
                 n_transitions,
                 n_trels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
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
                 n_decoder_layers=4,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        # create decoder for buffer front, stack top and rels
        self.transition_decoder = self.rel_decoder = None, None

        stack_size, buffer_size = [self.args.n_encoder_hidden//2] * 2 if (self.args.n_encoder_hidden % 2) == 0 \
            else [self.args.n_encoder_hidden//2, self.args.n_encoder_hidden//2+1]

        # create projection to reduce dimensionality of the encoder
        self.stack_proj = MLP(
                n_in=self.args.n_encoder_hidden, n_out=stack_size,
                dropout=mlp_dropout)
        self.buffer_proj = MLP(
            n_in=self.args.n_encoder_hidden, n_out=buffer_size,
            dropout=mlp_dropout
        )

        if self.args.decoder == 'lstm':
            decoder = lambda out_dim: DecoderLSTM(
                self.args.n_encoder_hidden, self.args.n_encoder_hidden, out_dim,
                self.args.n_decoder_layers, dropout=mlp_dropout, device=self.device
            )
        else:
            decoder = lambda out_dim: MLP(
                n_in=self.args.n_encoder_hidden, n_out=out_dim, dropout=mlp_dropout
            )

        self.transition_decoder = decoder(n_transitions)
        self.trel_decoder = decoder(n_trels)

        # create delay projection
        if self.args.delay != 0:
            self.delay_proj = MLP(n_in=self.args.n_encoder_hidden * (self.args.delay + 1),
                                  n_out=self.args.n_encoder_hidden, dropout=mlp_dropout)

        # create PoS tagger
        if self.args.encoder == 'lstm':
            self.pos_tagger = DecoderLSTM(
                self.args.n_encoder_hidden, self.args.n_encoder_hidden, self.args.n_tags, 
                num_layers=1, dropout=mlp_dropout, device=self.device
            )
        else:
            self.pos_tagger = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()

    def encoder_forward(self, words: torch.Tensor, feats: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Applies encoding forward pass. Maps a tensor of word indices (`words`) to their corresponding neural
        representation.
        Args:
            words: torch.IntTensor ~ [batch_size, bos + pad(seq_len) + eos + delay]
            feats: List[torch.Tensor]
            lens: List[int]

        Returns: x, qloss
            x: torch.FloatTensor ~ [batch_size, bos + pad(seq_len) + eos, embed_dim]
            qloss: torch.FloatTensor ~ 1

        """
        x = super().encode(words, feats)
        s_tag = self.pos_tagger(x[:, 1:-(1+self.args.delay), :])

        # adjust lengths to allow delay predictions
        # x ~ [batch_size, bos + pad(seq_len) + eos, embed_dim]
        if self.args.delay != 0:
            x = torch.cat([x[:, i:(x.shape[1] - self.args.delay + i), :] for i in range(self.args.delay + 1)], dim=2)
            x = self.delay_proj(x)

        # pass through vector quantization
        x, qloss = self.vq_forward(x)
        return x, s_tag, qloss

    def decoder_forward(self, x: torch.Tensor, stack_top: torch.Tensor, buffer_front: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            x: torch.FloatTensor ~ [batch_size, bos + pad(seq_len) + eos, embed_dim]
            stack_top: torch.IntTensor ~ [batch_size, pad(tr_len)]
            buffer_front: torch.IntTensor ~ [batch_size, pad(tr_len)]

        Returns: s_transition, s_trel
            s_transition: torch.FloatTensor ~ [batch_size, pad(tr_len), n_transitions]
            s_trel: torch.FloatTensor ~ [batch_size, pad(tr_len), n_trels]
        """
        batch_size = x.shape[0]

        # obtain encoded embeddings for stack_top and buffer_front
        stack_top = torch.stack([x[i, stack_top[i], :] for i in range(batch_size)])
        buffer_front = torch.stack([x[i, buffer_front[i], :] for i in range(batch_size)])

        # pass through projections
        stack_top = self.stack_proj(stack_top)
        buffer_front = self.buffer_proj(buffer_front)

        # stack_top ~ [batch_size, pad(tr_len), embed_dim//2]
        # buffer_front ~ [batch_size, pad(tr_len), embed_dim//2]
        # x ~ [batch_size, pad(tr_len), embed_dim]
        x = torch.concat([stack_top, buffer_front], dim=-1)

        # s_transition ~ [batch_size, pad(tr_len), n_transitions]
        # s_trel = [batch_size, pad(tr_len), n_trels]
        s_transition = self.transition_decoder(x)
        s_trel = self.trel_decoder(x)

        return s_transition, s_trel

    def forward(self, words: torch.Tensor, stack_top: torch.Tensor, buffer_front: torch.Tensor, feats: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Args:
            words: torch.IntTensor ~ [batch_size, bos + pad(seq_len) + eos + delay].
            stack_top: torch.IntTensor ~ [batch_size, pad(tr_len)]
            buffer_front: torch.IntTensor ~ [batch_size, pad(tr_len)]
            feats: List[torch.Tensor]

        Returns: s_transition, s_trel, qloss
            s_transition: torch.FloatTensor ~ [batch_size, pad(tr_len), n_transitions]
            s_trel: torch.FloatTensor ~ [batch_size, pad(tr_len), n_trels]
            qloss: torch.FloatTensor ~ 1
        """
        x, s_tag, qloss = self.encoder_forward(words, feats)
        s_transition, s_trel = self.decoder_forward(x, stack_top, buffer_front)
        return s_transition, s_trel, s_tag, qloss

    def decode(self, s_transition: torch.Tensor, s_trel: torch.Tensor, exclude: list = None):
        transition_preds = s_transition.argsort(-1, descending=True)
        if exclude:
            s_trel[:, :, exclude] = -1
        trel_preds = s_trel.argmax(-1)
        return transition_preds, trel_preds

    def loss(self, s_transition: torch.Tensor, s_trel: torch.Tensor, s_tag,
             transitions: torch.Tensor, trels: torch.Tensor, tags,
             smask: torch.Tensor, trmask: torch.Tensor, TRANSITION):
        s_transition, transitions = s_transition[trmask], transitions[trmask]
        s_trel, trels = s_trel[trmask], trels[trmask]

        # remove those values in trels that correspond to shift and reduce actions
        transition_pred = TRANSITION.vocab[s_transition.argmax(-1).flatten().tolist()]
        trel_mask = torch.tensor(list(map(lambda x: x not in ['reduce', 'shift'], transition_pred)))
        s_trel, trels = s_trel[trel_mask], trels[trel_mask]

        tag_loss = self.criterion(s_tag[smask], tags[smask]) if self.args.encoder == 'lstm' else torch.tensor(0).to(self.device)
        transition_loss = self.criterion(s_transition, transitions)
        trel_loss = self.criterion(s_trel, trels)

        return transition_loss + trel_loss + tag_loss
