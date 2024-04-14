import torch.nn as nn
from supar.modules.mlp import MLP
import torch

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float, device: str):
        super().__init__()

        self.input_size, self.hidden_size = input_size, hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=False,
                            dropout=dropout, batch_first=True)

        self.mlp = MLP(n_in=hidden_size, n_out=output_size, dropout=dropout,
                       activation=True)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: torch.Tensor [batch_size, seq_len, input_size]
        :returns torch.Tensor [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape

        # LSTM forward pass
        h0, c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device), \
                 torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        hn, cn = self.lstm(x, (h0, c0))

        # MLP forward pass
        output = self.mlp(hn.reshape(batch_size*seq_len, self.hidden_size))
        return output.reshape(batch_size, seq_len, self.output_size)

    def __repr__(self):
        return f'DecoderLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, output_size={self.output_size}'
