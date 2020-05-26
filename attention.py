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

class MultiHeadedLocationAwareAttention(nn.Module):
    r"""
    Multi-headed Location-Aware Attention
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output, align
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **align** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, in_features, num_heads=8, conv_out_channel=10):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv1d = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.q_proj = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.v_proj = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(self.dim, 1, bias=True)
        self.out_proj = nn.Linear(in_features << 1, in_features, bias=True)

    def forward(self, query, value, prev_align):  # value : BxTxD
        batch_size, seq_len = value.size(0), value.size(1)

        if prev_align is None:
            prev_align = value.new_zeros(batch_size, self.num_heads, seq_len)  # BxNxT

        score = self.get_attn_score(query, value, prev_align, batch_size, seq_len)
        align = F.softmax(score, dim=1)  # BNxT

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)  # BNxTxD

        context = torch.bmm(align.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)  # BNx1xT x BNxTxD => BNxD
        align = align.view(batch_size, self.num_heads, -1)  # BNxT => BxNxT

        combined = torch.cat([context, query], dim=2)
        output = self.out_proj(combined.view(-1, self.in_features << 1)).view(batch_size, -1, self.in_features)

        return output, align

    def get_attn_score(self, query, value, prev_align, batch_size, seq_len):
        loc_energy = torch.tanh(self.loc_proj(self.conv1d(prev_align).transpose(1, 2)))  # BxNxT => BxTxD
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)  # BxNxTxD => BNxTxD

        query = self.q_proj(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)  # BxNxTxD
        value = self.v_proj(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)  # BxNxTxD
        query = query.contiguous().view(-1, 1, self.dim)  # BNx1xD
        value = value.contiguous().view(-1, seq_len, self.dim)  # BNxTxD

        score = self.score_proj(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)  # BNxTxD => BNxT
        return score


class LocationAwareAttention(nn.Module):
    """ Implementation of Location-Aware Attention (Hybrid Attention) """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim, k=10, smoothing=True):
        super(LocationAwareAttention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.linear_q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.linear_v = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.linear_u = nn.Linear(k, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.linear_out = nn.Linear(attn_dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value, prev_align):
        batch_size, hidden_dim = query.size(0), query.size(2)

        conv_feat = torch.transpose(self.conv1d(prev_align.unsqueeze(1)), 1, 2)
        attn_score = self.linear_out(self.tanh(
                self.linear_q(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.linear_v(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.linear_u(conv_feat)
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            attn_score = torch.sigmoid(attn_score)
            align = torch.div(attn_score, attn_score.sum(dim=-1).unsqueeze(dim=-1))

        else:
            align = self.softmax(attn_score)

        context = torch.bmm(align.unsqueeze(dim=1), value).squeeze(dim=1)
        return context, align


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-product Attention

    Args:
        dim (int): dimention of attention

    Inputs: query, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, align
        - **context**: tensor containing the context vector from attention mechanism.
        - **align**: tensor containing the alignment from the encoder outputs.
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        attn_score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        align = self.softmax(attn_score)
        context = torch.bmm(align, value)
        return context, align


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        k (int): The dimension of convolution

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output, align
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **align** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, in_features, num_heads=8, k=10):
        super(CustomizingAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.linear_q = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.linear_v = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.linear_u = nn.Linear(k, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))
        self.linear_out = nn.Linear(in_features << 1, in_features, bias=True)
        self.normalize = nn.LayerNorm(in_features)

    def forward(self, query, value, prev_align):
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        loc_energy = self.get_loc_energy(prev_align, batch_size, v_len)  # get location energy

        query = self.linear_q(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.linear_v(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, align = self.scaled_dot(query, value)

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = self.normalize(self.linear_out(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output, align.squeeze()

    def get_loc_energy(self, prev_align, batch_size, v_len):
        conv_feat = self.conv1d(prev_align.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.linear_u(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention

    Applies a multi-headmechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        k (int): The dimension of convolution

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, in_features, num_head=8, dim=64):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_head = num_head
        self.dim = dim
        self.scaled_dot = ScaledDotProductAttention(dim)
        self.linear_q = nn.Linear(in_features, dim * num_head)
        self.linear_v = nn.Linear(in_features, dim * num_head)
        self.linear_out = nn.Linear(in_features << 1, in_features, bias=True)
        self.normalize = nn.LayerNorm(self.in_features)

    def forward(self, query, value):
        batch_size = value.size(0)
        residual = query

        query = self.linear_q(query).view(batch_size, -1, self.num_head, self.dim)
        value = self.linear_v(value).view(batch_size, -1, self.num_head, self.dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)

        context, _ = self.scaled_dot(query, value)
        context = context.view(self.num_head, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_head * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = self.normalize(self.linear_out(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)
        return output


class AdditiveAttention(nn.Module):
    """ Implementaion of Additive Attention (Bahdanau Attention) """
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        attn_score = self.linear_out(torch.tanh(self.linear_v(value) + self.linear_q(query.transpose(0, 1)))).squeeze(-1)
        align = nn.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(1), value)
        return context


class ContentBasedAttention(nn.Module):
    """ Implementation of Content-based Attention """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim):
        super(ContentBasedAttention, self).__init__()
        self.linear_q = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.linear_v = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.linear_out = nn.Linear(attn_dim, 1, bias=True)
        self.attn_dim = attn_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        batch_size, hidden_dim = query.size(0), query.size(2)

        attn_score = self.linear_out(self.tanh(
                self.linear_q(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.linear_v(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
                + self.bias
        )).squeeze(dim=-1)

        align = self.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(dim=1), value).squeeze(dim=1)
        return context


class DotProductAttention(nn.Module):
    """ Implementation of DotProduct Attention """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.normalize = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        attn_score = torch.bmm(query, value.transpose(1, 2))
        align = nn.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(align, value)

        combined = torch.cat((context, query), dim=2)
        output = self.normalize(self.linear_out(combined.view(-1, 2 * hidden_dim))).view(batch_size, -1, hidden_dim)

        return output
