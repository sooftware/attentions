"""
    @source_code{
        title={some attention implementation},
        author={Soohwan Kim},
        year={2020}
    }

    Pytorch Implementation of some attention
    any questions, bug reports or recommends, please Contacts sh951011@gmail.com
"""

import torch
import torch.nn as nn
import torch.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, n_head=4, dim=128):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.dim = dim
        self.W = nn.Linear(in_features, dim * n_head)
        self.V = nn.Linear(in_features, dim * n_head)
        self.fc = nn.Linear(in_features << 1, in_features)

    def forward(self, queries, values):
        batch_size = values.size(0)
        query_length = queries.size(1)
        value_length = values.size(1)

        preserved = queries

        queries = self.W(queries).view(batch_size, query_length, self.n_head, self.dim)
        values = self.V(values).view(batch_size, value_length, self.n_head, self.dim)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(-1, query_length, self.dim)
        values = values.permute(2, 0, 1, 3).contiguous().view(-1, value_length, self.dim)

        attn_score = torch.bmm(queries, values.transpose(1, 2))
        alignment = F.softmax(attn_score, dim=2)

        attn_val = torch.bmm(alignment, values).view(self.n_head, batch_size, query_length, self.dim)
        attn_val = attn_val.permute(1, 2, 0, 3).contiguous().view(batch_size, query_length, -1)

        combined = torch.cat([attn_val, preserved], dim=2)
        context = torch.tanh(self.fc(combined.view(-1, 2 * self.in_features))).view(batch_size, -1, self.in_features)

        return context


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, queries, values, inputs):
        queries = queries.transpose(0, 1)

        attn_score = self.fc(torch.tanh(self.V(values) + self.W(queries))).squeeze(-1)
        align = F.softmax(attn_score, -1)
        context = torch.bmm(align.unsqueeze(1), values)

        return context


class HybridAttention(nn.Module):
    def __init__(self, dec_hidden_dim, enc_hidden_dim, context_dim, k=10, smoothing=True):
        super(HybridAttention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.context_dim = context_dim
        self.smoothing = smoothing
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W = nn.Linear(dec_hidden_dim, context_dim, bias=False)
        self.V = nn.Linear(enc_hidden_dim, context_dim, bias=False)
        self.U = nn.Linear(k, context_dim, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(context_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, values, last_align):
        batch_size = queries.size(0)
        hidden_dim = queries.size(2)

        conv_attn = torch.transpose(self.loc_conv(last_align.unsqueeze(1)), 1, 2)
        attn_score = self.w(
            self.tanh(
                self.W(queries.reshape(-1, hidden_dim)).view(batch_size, -1, self.context_size)
                + self.V(values.reshape(-1, hidden_dim)).view(batch_size, -1, self.context_size)
                + self.U(conv_attn)
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
    def __init__(self, dec_hidden_dim, enc_hidden_dim, context_dim):
        super(ContentBasedAttention, self).__init__()
        self.W = nn.Linear(dec_hidden_dim, context_dim, bias=False)
        self.V = nn.Linear(enc_hidden_dim, context_dim, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_dim).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(context_dim, 1, bias=False)
        self.context_dim = context_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        attn_score = self.fc(
            self.tanh(
                self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_dim)
                + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_dim)
                + self.b
            )
        ).squeeze(dim=-1)
        align = self.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)

        return context


class DotProductAttention(nn.Module):
    def __init__(self, dec_hidden_dim):
        super(DotProductAttention, self).__init__()
        self.fc = nn.Linear(dec_hidden_dim * 2, dec_hidden_dim)

    def forward(self, queries, values):
        batch_size = queries.size(0)
        hidden_dim = queries.size(2)
        input_size = values.size(1)

        attn_score = torch.bmm(queries, values.transpose(1, 2))
        align = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(align, values)

        combined = torch.cat((context, queries), dim=2)
        context = torch.tanh(self.fc(combined.view(-1, 2 * hidden_dim))).view(batch_size, -1, hidden_dim)

        return context
