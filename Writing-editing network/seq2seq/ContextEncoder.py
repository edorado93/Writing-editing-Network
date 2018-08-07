import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
    def __init__(self, contextual_dim, number_of_contexts, word_embedding_dim):
        super(ContextEncoder, self).__init__()
        self.embedding = nn.Embedding(number_of_contexts, contextual_dim)
        self.transform = nn.Linear(contextual_dim, word_embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, topics, structure_abstracts):
        non_linearity1 = None
        non_linearity2 = None
        if topics is not None:
            embedded = self.embedding(topics)
            embedded = embedded.reshape(embedded.shape[0], 1, -1)
            non_linearity1 = self.tanh(embedded)
        if structure_abstracts is not None:
            structure_embedding = self.embedding(structure_abstracts)
            non_linearity2 = self.tanh(structure_embedding)
        return non_linearity1, non_linearity2