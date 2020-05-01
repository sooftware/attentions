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
        self.b = nn.Parameter(torch.FloatTensor(attn_dim).uniform_(-0.1, 0.1))
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
                + self.b
            )
        ).squeeze(dim=-1)

        align = self.softmax(attn_score)
        context = torch.bmm(align.unsqueeze(dim=1), V).squeeze(dim=1)

        return context


class DotProductAttention(nn.Module):
    def __init__(self, dec_hidden_dim):
        super(DotProductAttention, self).__init__()
        self.fc = nn.Linear(dec_hidden_dim * 2, dec_hidden_dim)

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
