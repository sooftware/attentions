"""
    @source_code{
        title={some attention implementation},
        author={Soohwan Kim},
        year={2020}
    }
"""

import torch
import torch.nn as nn


# Pytorch Implementation of some attention
# any questions, bug reports or recommends, please Contacts sh951011@gmail.com


class ScaledDotProductAttention(nn.Module):
    """ Implementation of Scaled Dot-product Attention """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        align = self.softmax(score)
        context = torch.bmm(align, value)
        return context, align


class MultiLocAwareAttention(nn.Module):
    r"""
    Multi-Head + Location-Aware Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        k (int): The dimension of convolution

    Inputs: query, value, prev_align
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """

    def __init__(self, in_features, num_heads=8, k=10):
        super(MultiLocAwareAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W_Q = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.W_V = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.W_U = nn.Linear(k, self.dim, bias=True)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.fc = nn.Linear(in_features << 1, in_features, bias=True)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, query, value, prev_align):  # (batch_size * num_heads, v_len)
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        loc_energy = self.get_loc_energy(prev_align, batch_size, v_len)

        q_s = self.W_Q(query).view(batch_size, q_len, self.num_heads * self.dim) + loc_energy
        v_s = self.W_V(value).view(batch_size, v_len, self.num_heads * self.dim)

        q_s = q_s.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        v_s = v_s.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)

        q_s = q_s.contiguous().view(-1, q_len, self.dim)  # (batch_size * num_heads, q_len, dim)
        v_s = v_s.contiguous().view(-1, v_len, self.dim)  # (batch_size * num_heads, v_len, dim)

        context, align = self.scaled_dot(q_s, v_s)
        context = context.view(self.num_heads, batch_size, q_len, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)
        align = align.squeeze()

        combined = torch.cat([context, residual], dim=2)
        output = self.norm(self.fc(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output, align

    def get_loc_energy(self, prev_align, batch_size, v_len):
        conv_feat = self.conv1d(prev_align.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.W_U(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, num_head=8, dim=64):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_head = num_head
        self.dim = dim
        self.scaled_dot = ScaledDotProductAttention(dim)
        self.W_Q = nn.Linear(in_features, dim * num_head)
        self.W_V = nn.Linear(in_features, dim * num_head)
        self.fc = nn.Linear(in_features + dim * num_head, in_features)
        self.norm = nn.LayerNorm(self.in_features)

    def forward(self, query, value):
        batch_size = value.size(0)
        residual = query

        q_s = self.W_Q(query).view(batch_size, -1, self.num_head, self.dim)
        v_s = self.W_V(value).view(batch_size, -1, self.num_head, self.dim)

        q_s = q_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)
        v_s = v_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)

        context, _ = self.scaled_dot(q_s, v_s)
        context = context.view(self.num_head, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_head * self.dim)
        combined = torch.cat([context, residual], dim=2)

        context = self.norm(self.fc(combined.view(-1, self.in_features + self.dim * self.num_head)))
        context = context.view(batch_size, -1, self.in_features)

        return context


class AdditiveAttention(nn.Module):
    """ Implementaion of Additive Attention (Bahdanau Attention) """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        query = query.transpose(0, 1)

        score = self.fc(torch.tanh(self.W_V(value) + self.W_Q(query))).squeeze(-1)
        align = nn.softmax(score)
        context = torch.bmm(align.unsqueeze(1), value)

        return context


class LocationAwareAttention(nn.Module):
    """ Implementation of Location-Aware Attention (Hybrid Attention) """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim, k=10, smoothing=True):
        super(LocationAwareAttention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W_Q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_U = nn.Linear(k, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(attn_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value, last_align):
        batch_size = query.size(0)
        hidden_dim = query.size(2)

        conv_feat = torch.transpose(self.conv(last_align.unsqueeze(1)), 1, 2)
        attn_score = self.fc(
            self.tanh(
                self.W_Q(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_V(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_U(conv_feat)
                + self.bias
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
    """ Implementation of Content-based Attention """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim):
        super(ContentBasedAttention, self).__init__()
        self.W_Q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(attn_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(attn_dim, 1, bias=False)
        self.attn_dim = attn_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        batch_size = query.size(0)
        hidden_dim = query.size(2)

        attn_score = self.fc(
            self.tanh(
                self.W_Q(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.W_V(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.bias
            )
        ).squeeze(dim=-1)

        align = self.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(dim=1), value).squeeze(dim=1)

        return context


class DotProductAttention(nn.Module):
    """ Implementation of DotProduct Attention """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, value):
        batch_size = query.size(0)
        hidden_dim = query.size(2)
        input_size = value.size(1)

        attn_score = torch.bmm(query, value.transpose(1, 2))
        align = nn.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn_val = torch.bmm(align, value)

        combined = torch.cat((attn_val, query), dim=2)
        context = self.norm(self.fc(combined.view(-1, 2 * hidden_dim))).view(batch_size, -1, hidden_dim)

        return context
