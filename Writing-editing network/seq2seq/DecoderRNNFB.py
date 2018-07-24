import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN
import random


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.hidden_size = hidden_size
        self.wrx = nn.Linear(hidden_size, hidden_size)
        self.wrh = nn.Linear(hidden_size, hidden_size)
        self.wix = nn.Linear(hidden_size, hidden_size)
        self.wih = nn.Linear(hidden_size, hidden_size)
        self.wnx = nn.Linear(hidden_size, hidden_size)
        self.wnh = nn.Linear(hidden_size, hidden_size)


    def forward(self, title, pg):

        r_gate = F.sigmoid(self.wrx(title) + self.wrh(pg))
        i_gate = F.sigmoid(self.wix(title) + self.wih(pg))
        n_gate = F.tanh(self.wnx(title) + torch.mul(r_gate, self.wnh(pg)))
        result =  torch.mul(i_gate, pg) + torch.mul(torch.add(-i_gate, 1), n_gate)
        return result


class DecoderRNNFB(BaseRNN):

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_GENERATED_STRUCTURE_LABELS = 'gen_labels'

    def __init__(self, vocab_size, embedding, max_len, embed_size,
                 sos_id, eos_id, n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, labels=None, context_model=None, use_labels=False, use_cuda=False):
        hidden_size = embed_size
        if bidirectional:
            hidden_size *= 2

        super(DecoderRNNFB, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.labels = labels
        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.embedding = embedding
        self.attention_title = Attention(self.hidden_size)
        self.attention_hidden = Attention(self.hidden_size)
        #self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc = Gate(self.hidden_size)
        self.out1 = nn.Linear(self.hidden_size, self.output_size)
        self.out2 = nn.Linear(self.hidden_size, self.output_size)

        # Structural embedding variables requirement during test time
        self.context_model = context_model
        self.use_labels = use_labels
        self.use_cuda = use_cuda


    def forward_step(self, input_var, pg_encoder_states, hidden, encoder_outputs, topical_embedding=None, structural_embedding=None):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        if topical_embedding is not None:
            embedded = torch.cat([topical_embedding.expand(-1, embedded.shape[1], -1), embedded], dim=2)
        if structural_embedding is not None:
            embedded = torch.cat((embedded, structural_embedding), dim=2)
        embedded = self.input_dropout(embedded)

        attn = None

        output_states, hidden = self.rnn(embedded, hidden)
        output_states_attn1, attn1 = self.attention_title(output_states, encoder_outputs)
        if pg_encoder_states is None:
            output_states_attn = output_states_attn1
        else:
            output_states_attn2, attn2 = self.attention_hidden(output_states, pg_encoder_states)
            output_states_attn = self.fc(output_states_attn1, output_states_attn2)
        outputs = self.out1(output_states_attn.view(-1, self.hidden_size)).\
            view(batch_size, output_size, -1)
        return outputs, output_states_attn, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                pg_encoder_states=None, function=F.log_softmax, teacher_forcing_ratio=0, topical_embedding=None, structural_embedding=None):
        ret_dict = dict()
        ret_dict[DecoderRNNFB.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs,
                                    encoder_hidden, encoder_outputs,
                                    function, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if teacher_forcing_ratio == 1 else False

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            structural_embedding = structural_embedding[:, :-1]
            decoder_outputs, decoder_output_states, decoder_hidden, attn = \
                self.forward_step(decoder_input, pg_encoder_states,
                                decoder_hidden, encoder_outputs, topical_embedding, structural_embedding)
        else:
            decoder_outputs = []
            decoder_output_states = []
            sequence_symbols = []
            generated_structure_labels = []
            lengths = np.array([max_length] * batch_size)

            # 30 percent of the times select a random word, otherwise
            # select a word that doesn't occur in the last window of 5 words. This
            # if to avoid repetition and introduce some randomness.
            def penalise_repetitions(step_output):
                output = []
                if random.random() < 0.3:
                    word_weights = step_output.squeeze().data.div(1.).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)
                    return word_idx.view(step_output.shape[0], 1).cuda() if self.use_cuda else word_idx.view(step_output.shape[0], 1)
                symbols = step_output.topk(10)[1]
                for i, s in enumerate(symbols):
                    previous_window = [t[i] for t in sequence_symbols[-5:]]
                    use_first = True
                    for sym in s:
                        if sym not in previous_window:
                            output.append(sym)
                            use_first = False
                            break
                    if use_first:
                        output.append(s[0])

                output = torch.stack(output).unsqueeze(1)
                return output.cuda() if self.use_cuda else output

            def decode(step, step_output, step_output_state=None, step_attn=None):
                if step_output_state is not None:
                    decoder_outputs.append(step_output)
                    decoder_output_states.append(step_output_state)
                ret_dict[DecoderRNNFB.KEY_ATTN_SCORE].append(step_attn)
                symbols = penalise_repetitions(step_output)#step_output.topk(1)[1]
                sequence_symbols.append(symbols)
                eos_batches = symbols.data.eq(self.eos_id)
                if eos_batches.dim() > 0:
                    eos_batches = eos_batches.cpu().view(-1).numpy()
                    update_idx = ((lengths > step) & eos_batches) != 0
                    lengths[update_idx] = len(sequence_symbols)
                return symbols

            decoder_input = inputs[:, 0].unsqueeze(1)
            structural_embedding =  None
            gen_label = None
            for di in range(max_length):
                structural_embedding, gen_label = self._get_new_structure_label(decoder_input, structural_embedding, inputs.shape[0], gen_label)
                generated_structure_labels.append(gen_label)
                decoder_output, decoder_output_state, decoder_hidden, step_attn = \
                    self.forward_step(decoder_input, pg_encoder_states, decoder_hidden,
                                      encoder_outputs, topical_embedding, structural_embedding)
                # # not allow decoder to output UNK
                decoder_output[:, :, 3] = -float('inf')

                step_output = decoder_output.squeeze(1)
                step_output_state = decoder_output_state.squeeze(1)
                symbols = decode(di, step_output, step_output_state, step_attn)
                decoder_input = symbols
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            decoder_output_states = torch.stack(decoder_output_states, dim=1)
            ret_dict[DecoderRNNFB.KEY_SEQUENCE] = sequence_symbols
            ret_dict[DecoderRNNFB.KEY_LENGTH] = lengths.tolist()
            ret_dict[DecoderRNNFB.KEY_GENERATED_STRUCTURE_LABELS] = generated_structure_labels

        return decoder_outputs, decoder_output_states, ret_dict

    #TODO Figure out how to make this work for a batch of abstracts.
    def _get_new_structure_label(self, symbol_output, current_structure, batch_size, current_label):

        if not self.use_labels:
            return None

        is_changed = False
        if current_structure is None:
            current_structure = torch.LongTensor([self.labels["introduction"]]).expand(batch_size, 1)
            is_changed = True
        else:
            # This means that the sentence has ended and we have to resample
            if symbol_output.item() == self.labels["full_stop"] or symbol_output.item() == self.labels["question_mark"]:
                random_sample = random.random()
                if current_label == self.labels["introduction"]:
                    if random_sample >= 0.99:
                        current_structure = torch.LongTensor([self.labels["conclusion"]]).expand(batch_size, 1)
                        is_changed = True
                    elif random_sample >= 0.51:
                        current_structure = torch.LongTensor([self.labels["body"]]).expand(batch_size, 1)
                        is_changed = True
                else:
                    if random_sample >= 0.745:
                        current_structure = torch.LongTensor([self.labels["conclusion"]]).expand(batch_size, 1)
                        is_changed = True


        if is_changed:
            #TODO: Won't work for a batch of sentences. Need to change this.
            gen_label = current_structure[0].item()
            if self.use_cuda: current_structure = current_structure.cuda()
            _, structural_embedding = self.context_model(None, current_structure)
        else:
            structural_embedding = current_structure
            gen_label = current_label
        return structural_embedding, gen_label

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if self.use_cuda: inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1   

        return inputs, batch_size, max_length
