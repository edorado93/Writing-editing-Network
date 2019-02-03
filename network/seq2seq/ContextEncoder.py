import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
    def __init__(self, contextual_dim, number_of_contexts):
        super(ContextEncoder, self).__init__()
        self.contextual_embedding = nn.Embedding(number_of_contexts, contextual_dim)
        self.structural_embedding = nn.Embedding(3, 2)
        self.tanh = nn.Tanh()

    def forward(self, topics, structure_abstracts):
        non_linearity1 = None
        non_linearity2 = None
        if topics is not None:
            embedded = self.contextual_embedding(topics)
            embedded = embedded.reshape(embedded.shape[0], 1, -1)
            non_linearity1 = self.tanh(embedded)
        if structure_abstracts is not None:
            structure_embedding = self.structural_embedding(structure_abstracts)
            non_linearity2 = self.tanh(structure_embedding)
        return non_linearity1, non_linearity2