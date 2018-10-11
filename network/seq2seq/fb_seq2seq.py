import torch.nn as nn
import torch.nn.functional as F
import torch

class FbSeq2seq(nn.Module):

    def __init__(self, encoder_title, encoder, context_encoder, decoder, decode_function=F.log_softmax):
        super(FbSeq2seq, self).__init__()
        self.context_encoder = context_encoder
        self.encoder_title = encoder_title
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, prev_generated_seq=None, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, topics=None, structure_abstracts=None):

        if topics is not None or structure_abstracts is not None:
            topical_embedding, structural_embedding = self.context_encoder(topics, structure_abstracts)
        else:
            topical_embedding, structural_embedding = None, None

        # We don't have to pass the structure embedding to the title encoder. The structure is only relevant for the abstract
        encoder_outputs, encoder_hidden = self.encoder_title(input_variable, input_lengths, topical_embedding=topical_embedding)
        if prev_generated_seq is None:
            pg_encoder_states = None
        else:
            pg_structural_embedding = None
            if structural_embedding is not None:
                # The generated sequence has length one shorter than the original abstract. Hence we have to change the structural embedding as well.
                # The decoder doesn't generate the first word of the abstract. Hence, we ignore the label for the first word during TRAINING.
                # During testing, we have the number of structure labels equal to the length of the generated sequence. No need to trim then
                if structural_embedding.shape[1] != prev_generated_seq.shape[1]:
                    pg_structural_embedding = structural_embedding[:, 1:]
                else:
                    pg_structural_embedding = structural_embedding
            pg_encoder_states, pg_encoder_hidden = self.encoder(prev_generated_seq, topical_embedding=topical_embedding, structural_embedding=pg_structural_embedding)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              pg_encoder_states=pg_encoder_states,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              topical_embedding=topical_embedding,
                              structural_embedding=structural_embedding)

        loss = None
        if target_variable is not None:
            decoder_outputs_reshaped = result[0].view(-1, self.encoder.embedding.num_embeddings)
            target_variables_reshaped = target_variable[:, 1:].contiguous().view(-1)
            loss = self.criterion(decoder_outputs_reshaped, target_variables_reshaped)
            loss = loss.unsqueeze(0)
        # Current output of the model. This will be the previously generated abstract for the model.
        prev_generated_seq = torch.squeeze(torch.topk(result[0], 1, dim=2)[1]).view(-1, result[0].size(1))

        return loss, prev_generated_seq, result[2]
