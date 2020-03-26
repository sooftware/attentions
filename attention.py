"""
MIT License

Copyright (c) 2020 KimSooHwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.functional as F


# Pytorch Implementation of Some Attention
# any questions, bug reports or recommends, please Contacts sh951011@gmail.com


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, n_head=4, dim=128):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.out = nn.Linear(in_features, in_features)
        self.n_head = n_head
        self.linear_q = nn.Linear(in_features, dim * n_head)
        self.linear_k = nn.Linear(in_features, dim * n_head)
        self.dim = dim


    def forward(self, query, key):
        batch_size = key.size(0)
        query_length = query.size(1)
        key_length = key.size(1)

        residual = query

        query = self.linear_q(query).view(batch_size, query_length, self.n_head, self.dim).permute(2, 0, 1, 3)
        key = self.linear_k(key).view(batch_size, key_length, self.n_head, self.dim).permute(2, 0, 1, 3)

        query = query.contiguous().view(-1, query_length, self.dim)  # -1 = n_head * batch_size
        key = key.contiguous().view(-1, key_length, self.dim)

        # get attention score
        attn_score = torch.bmm(query, key.transpose(1, 2))

        # get attention distribution
        attn_distribution = F.softmax(attn_score, dim=2)

        # get context vector
        context = torch.bmm(attn_distribution, key).view(self.n_head, batch_size, query_length, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, query_length, -1)

        # concatenate context & query
        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.out(combined.view(-1, 2 * self.in_features))).view(batch_size, -1, self.in_features)

        return output


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_output, inputs):
        decoder_output = decoder_output.transpose(0, 1)

        attn_score = self.linear_out(torch.tanh(self.linear_k(encoder_outputs) + self.linear_q(decoder_output)))
        attn_score = attn_score.squeeze(-1)

        attn_distribution = F.softmax(attn_score, -1)

        context = attn_distribution.unsqueeze(1).bmm(encoder_outputs)

        return context



class HybridAttention(nn.Module):
    def __init__(self, decoder_hidden_size, encoder_hidden_size, context_size, k = 10, smoothing=True):
        super(HybridAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.context_size = context_size
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W = nn.Linear(decoder_hidden_size, context_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, context_size, bias=False)
        self.U = nn.Linear(k, context_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(context_size, 1, bias=False)
        self.smoothing = smoothing
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs, last_align):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        if last_align is None:
            attn_scores = self.w(
                self.tanh(
                    self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.b
                )
            ).squeeze(dim=-1)
        else:
            conv_prev_align = torch.transpose(self.loc_conv(last_align.unsqueeze(1)), 1, 2)
            attn_scores = self.w(
                self.tanh(
                    self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.U(conv_prev_align)
                    + self.b
                )
            ).squeeze(dim=-1)

        if self.smoothing:
            attn_scores = torch.sigmoid(attn_scores)
            align = torch.div(attn_scores, attn_scores.sum(dim=-1).unsqueeze(dim=-1))
        else:
            align = self.softmax(attn_scores)

        context = torch.bmm(align.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)

        return context, align


class ContentBasedAttention(nn.Module):
    def __init__(self, decoder_hidden_size, encoder_hidden_size, context_size):
        super(ContentBasedAttention, self).__init__()
        self.W = nn.Linear(decoder_hidden_size, context_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, context_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(context_size, 1, bias=False)
        self.context_size = context_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        attn_scores = self.w(
            self.tanh(
                self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                + self.b
            )
        ).squeeze(dim=-1)
        align = self.softmax(attn_scores)
        context = torch.bmm(align.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)

        return context


class DotAttention(nn.Module):
    def __init__(self, decoder_hidden_size):
        super(DotAttention, self).__init__()
        self.linear_out = nn.Linear(decoder_hidden_size*2, decoder_hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        input_size = encoder_outputs.size(1)

        # get attention score
        attn_score = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        # get attention distribution
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # get attention value
        attn_val = torch.bmm(attn_distribution, encoder_outputs) # get attention value
        # concatenate attn_val & decoder_output
        combined = torch.cat((attn_val, decoder_output), dim=2)
        context = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return context