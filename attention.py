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
    r"""
    Applies an multi-head attention mechanism on the output features from the decoder.
    Refer to 「State-of-the-art Speech Recognition With Sequence-to-Sequence Models」 Paper
    https://arxiv.org/abs/1712.01769
    Args:
        decoder_hidden_size (int): The number of expected features in the output
    Inputs: decoder_output, encoder_outputs
        - **decoder_output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **encoder_outputs** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    Examples::
        >>> attention = MultiHeadAttention(hidden_size, n_head=4, dim=128)
        >>> output = attention(decoder_output, encoder_outputs)
    """

    def __init__(self, hidden_size, n_head=4, dim=128):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
        self.dim = dim
        self.n_head = n_head
        self.linear_q = nn.Linear(hidden_size, dim * n_head)
        self.linear_k = nn.Linear(hidden_size, dim * n_head)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        dec_len = decoder_output.size(1)
        enc_len = encoder_outputs.size(1)

        query = self.linear_q(decoder_output).view(batch_size, dec_len, self.n_head, self.dim).permute(2, 0, 1, 3)
        key = self.linear_k(encoder_outputs).view(batch_size, enc_len, self.n_head, self.dim).permute(2, 0, 1, 3)

        query = query.contiguous().view(-1, dec_len, self.dim)  # -1 = n_head * batch_size
        key = key.contiguous().view(-1, enc_len, self.dim)

        # get attention score
        attn_score = torch.bmm(query, key.transpose(1, 2))

        # get attention distribution
        attn_distribution = F.softmax(attn_score, dim=2)

        # get context vector
        context = torch.bmm(attn_distribution, key).view(self.n_head, batch_size, dec_len, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, dec_len, -1)

        # concatenate attn_val & decoder_output
        combined = torch.cat((context, decoder_output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(batch_size, -1,
                                                                                           self.hidden_size)

        return output


class HybridAttention(nn.Module):
    '''
    Score function : Hybrid attention (Location-aware Attention)

    .. math ::
        score = w^T( tanh( Ws + Vhs + Uf + b ) )
            => s : decoder_output
               hs : encoder_outputs
               f : loc_conv(last_alignment)
               b : bias

    Reference:
        「Attention-Based Models for Speech Recognition」 Paper
         https://arxiv.org/pdf/1506.07503.pdf
    '''
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

    def forward(self, decoder_output, encoder_outputs, last_alignment):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        if last_alignment is None:
            attn_scores = self.w(
                self.tanh(
                    self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.b
                )
            ).squeeze(dim=-1)
        else:
            conv_prev_align = torch.transpose(self.loc_conv(last_alignment.unsqueeze(1)), 1, 2)
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
            alignment = torch.div(attn_scores, attn_scores.sum(dim=-1).unsqueeze(dim=-1))
        else:
            alignment = self.softmax(attn_scores)

        context = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)
        return context, alignment


class ContentBasedAttention(nn.Module):
    """
    Applies an content-based attention mechanism on the output features from the decoder.

    Reference:
        「Attention-Based Models for Speech Recognition」 Paper
         https://arxiv.org/pdf/1506.07503.pdf

    """
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
        alignment = self.softmax(attn_scores)
        context = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)

        return context


class DotAttention(nn.Module):
    """
    Applies an dot product attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * encoder_output) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: decoder_output, encoder_output
        - **decoder_output** (batch, output_len, hidden_size): tensor containing the output features from the decoder.
        - **encoder_output** (batch, input_len, hidden_size): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """
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