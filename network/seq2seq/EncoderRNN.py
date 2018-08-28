import torch.nn as nn
import torch
import numpy as np

from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embedding, max_len, input_length, hidden_size,
                input_dropout_p=0, dropout_p=0, n_layers=1,
                bidirectional=False, rnn_cell='gru', variable_lengths=True): 
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = embedding
        self.rnn = self.rnn_cell(input_length, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None, topical_embedding=None, structural_embedding=None):

        if input_lengths is not None:
            input_lengths = input_lengths.tolist()

        embedded = self.embedding(input_var)
        if topical_embedding is not None:
            embedded = torch.cat([topical_embedding.expand(-1, embedded.shape[1], -1), embedded], dim=2)
        if structural_embedding is not None:
            embedded = torch.cat((embedded, structural_embedding), dim=2)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
