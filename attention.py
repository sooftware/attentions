"""
    @source_code{
        title={some attention implementation},
        author={Soohwan Kim},
        year={2020}
    }
"""

import torch
import torch.nn as nn
import torch.functional as F


# Pytorch Implementation of some attention
# any questions, bug reports or recommends, please Contacts sh951011@gmail.com


import torch
import torch.nn as nn
import numpy as np


class MultiHeadLocationAwareAttention(nn.Module):
    r"""
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: 4)
        k (int): The dimension of convolution
        smoothing (bool): Replace the unbounded exponential function of the softmax function
        with the bounded logistic sogmoid.

    Inputs: Q, V, last_align
        - **Q**: tensor containing the output features from the decoder.
        - **V**: tensor containing features of the encoded input sequence.
        - **last_align** : tensor containing previous timestep`s alignment

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """

    def __init__(self, in_features, num_heads=4, k=10, smoothing=True):
        super(MultiHeadLocationAwareAttention, self).__init__()
        self.smoothing = smoothing
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W_Q = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.W_V = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.W_U = nn.Linear(k, self.dim * num_heads, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(self.dim * num_heads).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(in_features << 1, in_features, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, V, last_align):
        batch_size = V.size(0)
        q_len = Q.size(1)
        v_len = V.size(1)

        residual = Q
        U = torch.transpose(self.conv(last_align.unsqueeze(1)), 1, 2)

        q_s = self.W_Q(Q).view(batch_size, q_len, self.num_heads * self.dim)
        v_s = self.W_V(V).view(batch_size, v_len, self.num_heads * self.dim) + self.W_U(U) + self.bias

        q_s = q_s.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        v_s = v_s.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)

        q_s = q_s.contiguous().view(-1, q_len, self.dim)  # (batch_size * num_heads, q_len, dim)
        v_s = v_s.contiguous().view(-1, v_len, self.dim)  # (batch_size * num_heads, v_len, dim)

        score = torch.bmm(q_s, v_s.transpose(1, 2)) / np.sqrt(self.dim)  # scaled dot-product

        if self.smoothing:
            score = torch.sigmoid(score)
            align = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))

        else:
            align = self.softmax(score)

        context = torch.bmm(align, v_s).view(self.num_heads, batch_size, q_len, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.fc(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output, align


class MultiHeadAttention(nn.Module):

    def __init__(self, in_features, num_head=8, dim=64):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_head = num_head
        self.dim = dim
        self.W_Q = nn.Linear(in_features, dim * num_head)
        self.W_V = nn.Linear(in_features, dim * num_head)
        self.fc = nn.Linear(in_features + dim * num_head, in_features)

    def forward(self, Q, V):
        batch_size = V.size(0)
        residual = Q

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_head, self.dim)
        v_s = self.W_V(V).view(batch_size, -1, self.num_head, self.dim)

        q_s = q_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)
        v_s = v_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)

        attn_score = torch.bmm(q_s, v_s.transpose(1, 2))
        align = F.softmax(attn_score, dim=2)

        attn_val = torch.bmm(align, v_s).view(self.num_head, batch_size, -1, self.dim)
        attn_val = attn_val.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_head * self.dim)
        combined = torch.cat([attn_val, residual], dim=2)

        context = torch.tanh(self.fc(combined.view(-1, self.in_features + self.dim * self.num_head)))
        context = context.view(batch_size, -1, self.in_features)

        return context


class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, Q, V, inputs):
        Q = Q.transpose(0, 1)

        attn_score = self.fc(torch.tanh(self.W_V(V) + self.W_Q(Q))).squeeze(-1)
        align = F.softmax(attn_score, -1)
        context = torch.bmm(align.unsqueeze(1), V)

        return context


class LocationAwareAttention(nn.Module):
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim, k=10, smoothing=True):
        super(LocationAwareAttention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W_Q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_U = nn.Linear(k, attn_dim, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(attn_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, V, last_align):
        batch_size = Q.size(0)
        hidden_dim = Q.size(2)

        U = torch.transpose(self.conv(last_align.unsqueeze(1)), 1, 2)
        attn_score = self.fc(
            self.tanh(
                self.W_Q(Q.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_V(V.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_U(U)
                + self.b
            )
        ).squeeze(dim=-1)

        if self.smoothing:
            attn_score = torch.sigmoid(attn_score)
            align = torch.div(attn_score, attn_score.sum(dim=-1).unsqueeze(dim=-1))

        else:
            align = self.softmax(attn_score)

        context = torch.bmm(align.unsqueeze(dim=1), values).squeeze(dim=1)
        return context, align


class ContentBasedAttention(nn.Module):

    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim):
        super(ContentBasedAttention, self).__init__()
        self.W_Q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(attn_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(attn_dim, 1, bias=False)
        self.attn_dim = attn_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, V):
        batch_size = Q.size(0)
        hidden_dim = Q.size(2)

        attn_score = self.fc(
            self.tanh(
                self.W_Q(Q.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_V(V.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.bias
            )
        ).squeeze(dim=-1)

        align = self.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(dim=1), V).squeeze(dim=1)

        return context


class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, Q, V):
        batch_size = Q.size(0)
        hidden_dim = Q.size(2)
        input_size = V.size(1)

        attn_score = torch.bmm(Q, V.transpose(1, 2))
        align = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn_val = torch.bmm(align, V)

        combined = torch.cat((attn_val, Q), dim=2)
        context = torch.tanh(self.fc(combined.view(-1, 2 * hidden_dim))).view(batch_size, -1, hidden_dim)

        return context
