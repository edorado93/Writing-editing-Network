import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
    def __init__(self, contextual_dim, number_of_contexts, word_embedding_dim):
        super(ContextEncoder, self).__init__()
        self.embedding = nn.Embedding(number_of_contexts, contextual_dim)
        self.transform = nn.Linear(contextual_dim, word_embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, topics):
        embedded = self.embedding(topics)
        embedded = embedded.reshape(embedded.shape[0], 1, -1)
        non_linearity = self.tanh(embedded)
        return non_linearity