import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        mask = torch.eq(attn, 0).data.byte()
        attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)


        if not output.is_contiguous():
            output = output.contiguous()

        return output, attn

class IntraAttention(nn.Module):
    def __init__(self, dim):
        super(IntraAttention, self).__init__()
        self.w_dec_attn = nn.init.normal_(torch.empty(dim, dim))
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, hidden_states):
        input_size = hidden_states.size(1)
        batch_size = hidden_states.size(0)
        hidden_size = hidden_states.size(2)
        if hidden_states.is_cuda:
            self.w_dec_attn = self.w_dec_attn.cuda()
        context_vectors = [hidden_states[:, 0, :]]
        for i in range(1, input_size):
            # [ batch_size * hidden_size ] = ( B * H )
            current_hidden_state = hidden_states[:, i, :]

            # [ batch_size * len(decoder_states), hidden_size ] = ( B * L * H ) -----> ( B, H, L )
            prev_hidden_states = hidden_states[:, :i, :].transpose(1, 2)

            # [ batch_size * hidden_size ] = ( B * H ) -----> ( B, 1, H )
            _dec_attn = torch.mm(current_hidden_state.squeeze(1), self.w_dec_attn).unsqueeze(1)

            # [ batch_size * 1 * len(decoder_states) ] -----> (B, L)
            # This value corresponds to equation 6 in the paper.
            attention_score_unnormalized = torch.bmm(_dec_attn, prev_hidden_states).squeeze(1)

            # [ batch_size * len(decoder_states) ] = ( B * L ) -----> ( B, L, 1)
            # This value corresponds to equation 7 in the paper.
            normalized_scores = self.softmax(attention_score_unnormalized).unsqueeze(2)

            # We do an element wise product of ( B, L, H ) and ( B, L, 1 ).
            # Broadcasting will kick in here.
            # Finally we do the summation. Equation 8 in the paper. -----> ( B, H )
            context = torch.sum(prev_hidden_states.transpose(1, 2) * normalized_scores, dim=1)

            # Record the modified context vectors for intra-attention
            context_vectors.append(context)

        # ( No. of Hidden states * batch_size *  batch_size * hidden_size = ( W * B * H )-----> ( B * W * H )
        context_vectors = torch.stack(context_vectors).t()

        # ************************************************  COMBINE THE ATTENTION VECTORS WITH THE HIDDEN LAYERS

        # concat -> (batch, out_len, 2*dim) -----> ( B * W * 2H )
        combined = torch.cat((hidden_states, context_vectors), dim=2)

        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        if not output.is_contiguous():
            output = output.contiguous()

        return output
