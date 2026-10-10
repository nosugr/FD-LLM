"""LSTM feature extraction for FD-LLM data-text alignment (Section 3.1)."""

import torch
from torch import nn


class LSTMDataEncoder(nn.Module):
    """Encode preprocessed vibration windows into a shared embedding space.

    Defaults map (batch, 1024, 1) to (batch, 4096). Layer count, hidden size,
    dropout and final-state pooling are implementation choices, not parameters
    specified by the paper. With bidirectional=True, concatenate both final
    hidden states of the last layer before projection.

    Inputs must be floating-point tensors matching the model's device/dtype.
    Normalization/windowing belongs to preprocessing; embedding normalization,
    contrastive objectives and training belong to the caller.
    """

    def __init__(
        self,
        *,
        input_size: int = 1,
        sequence_length: int = 1024,
        hidden_size: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 4096,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (("input_size", input_size),
                            ("sequence_length", sequence_length),
                            ("hidden_size", hidden_size),
                            ("num_layers", num_layers),
                            ("embedding_dim", embedding_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        if type(bidirectional) is not bool:
            raise ValueError("bidirectional must be a boolean")

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # LSTM 的 dropout 仅作用于层间，单层时需关闭。
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.projection = nn.Linear(hidden_size * (2 if bidirectional else 1),
                                    embedding_dim)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return unnormalized embeddings, with fresh zero states per call."""
        if signals.ndim != 3 or signals.shape[1:] != (
            self.sequence_length, self.input_size
        ) or signals.shape[0] == 0:
            raise ValueError(
                f"signals must have shape (batch_size > 0, "
                f"{self.sequence_length}, {self.input_size}); "
                f"got {tuple(signals.shape)}"
            )
        if not signals.is_floating_point():
            raise TypeError("signals must be a floating-point tensor")

        _, (hidden, _) = self.lstm(signals)
        # 取最后一层的最终状态；双向时拼接两个方向，再投影到文本特征维度。
        if self.lstm.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            features = hidden[-1]
        return self.projection(features)
