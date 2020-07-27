import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from dot import ScaledDotProductAttention


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The dimension of convolution

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, conv_out_channel: int = 10) -> None:
        super(CustomizingAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(1, conv_out_channel, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size * self.num_heads, v_len)

        loc_energy = self.get_loc_energy(last_attn, batch_size, v_len)  # get location energy

        query = self.query_proj(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.value_proj(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, attn = self.scaled_dot_attn(query, value)
        attn = attn.squeeze()

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        return context, attn

    def get_loc_energy(self, last_attn: Tensor, batch_size: int, v_len: int) -> Tensor:
        conv_feat = self.conv1d(last_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc_proj(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy
