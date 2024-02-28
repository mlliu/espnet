#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from typing import Tuple
from typing import Dict, List, Optional, Tuple, Union

import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Embedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        assert check_argument_types()
        super().__init__()
        self.embed_dim = embed_dim
        # TODO(sdalmia): check for padding idx
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, embed_dim),
            pos_enc_class(embed_dim, positional_dropout_rate),
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        x = self.embed(input)

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim



class Embedding_multi_input(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    def __init__(
        self,
        # the input_size is a list with 4 elements, some elements are None
        input_size: List[int] = [400, 400, None, None],
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        #assert check_argument_types()
        super().__init__()
        self.embed_dim = embed_dim
        # TODO(sdalmia): check for padding idx
        # self.embed = torch.nn.Sequential(
        #     torch.nn.Embedding(input_size, embed_dim),
        #     pos_enc_class(embed_dim, positional_dropout_rate),
        # )
        # self.embed1 = torch.nn.Embedding(input_size[0], embed_dim)
        # self.embed2 = torch.nn.Embedding(input_size[1], embed_dim)
        # self.pos_enc = pos_enc_class(embed_dim, positional_dropout_rate)
        self.num_inputs = 0
        for i, size in enumerate(input_size):
            if size is not None:
                setattr(self, f'embed{i+1}', torch.nn.Embedding(size, embed_dim))
                self.num_inputs += 1
        # self.embed_1 = torch.nn.Embedding(input_size[0], embed_dim)
        # self.embed_2 = torch.nn.Embedding(input_size[1], embed_dim)
        self.linear = torch.nn.Linear(self.num_inputs*embed_dim, embed_dim)
        # self.linear = torch.nn.Linear(2*embed_dim, embed_dim)
        self.pos_enc = pos_enc_class(embed_dim, positional_dropout_rate)


    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        #x = self.embed(input)
        # input shape is (B, T, 2)
        # x1 = self.embed1(input[:,:,0])
        # x2 = self.embed2(input[:,:,1])
        # x = torch.cat((x1, x2), dim=2) # (B, T, 2*D)
        x = []
        for i in range(self.num_inputs):
            x.append(getattr(self, f'embed{i+1}')(input[:,:,i]))

        # to save memory, we can use the following line instead of the for loop
        #x = [getattr(self, f'embed{i+1}')(input[:,:,i]) for i in range(self.num_inputs)]

        # after that, we can delete the input 'input' to save memory
        # del input
        #
        x = torch.cat(x, dim=2)  # (B, T, self.num_inputs*D)

        # add a linear projection layer, to project the input x to (B, T, D)
        x = self.linear(x)

        x = self.pos_enc(x)

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim

