import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, use_cuda=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        h, c = torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim)
        if self.use_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, abstract, batch_size):
        h0, c0 = self.init_hidden(batch_size)
        embeds = self.word_embeddings(abstract)
        lstm_out, self.hidden = self.lstm(embeds, (h0, c0))
        return lstm_out, self.hidden

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, batch_size):
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out((output.view(batch_size, 1, -1)))
        output = output.squeeze(2)
        return output, hidden

def decoder_train(input_tensor, encoder_hidden, decoder, seq_length, batch_size, use_cuda):
    decoder_input = torch.zeros((batch_size, 1), dtype=torch.long)
    decoder_output_f = torch.tensor([])
    if use_cuda:
        decoder_input = decoder_input.cuda()
        decoder_output_f = decoder_output_f.cuda()
    decoder_hidden = encoder_hidden

    for di in range(seq_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
        decoder_input = input_tensor[:, di]
        decoder_output_f = torch.cat((decoder_output_f, decoder_output), dim=1)
    return decoder_output_f

# critic
class Critic(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, use_cuda=False):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, 1)
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        h, c = torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim)
        if self.use_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, abstract, batch_size):
        h0, c0 = self.init_hidden(batch_size)
        embeds = self.word_embeddings(abstract)
        lstm_out, self.hidden = self.lstm(embeds, (h0, c0))
        out_space = self.hidden2tag(lstm_out.view(batch_size, len(abstract[0]), -1).squeeze(2))
        out_space1 = out_space.squeeze(2)
        return out_space1

def criticLoss(dis_pred, est_val, seq_length, batch_size, config, use_cuda=False):
    gamma = 0.8835659
    m = nn.Sigmoid()
    rewards = torch.log(m(dis_pred))
    cumulative_rewards = []

    rewards = torch.unbind(rewards, dim=1)
    for t in range(seq_length):
        cum_value = torch.zeros(batch_size)
        if use_cuda:
            cum_value = cum_value.cuda()
        for s in range(t, seq_length):
            discount = (gamma ** (s - t))
            cum_value += discount * rewards[s]
        cumulative_rewards.append(cum_value)
    cumulative_rewards = torch.stack(cumulative_rewards, dim=1)

    loss = nn.MSELoss()
    cumulative_rewards = cumulative_rewards.detach()
    critic_loss = loss(est_val, cumulative_rewards)
    return critic_loss

def reinforce(gen_log_prob, dis_pred, est_val, seq_length, batch_size, config, use_cuda=False):
    gamma = 0.8835659
    m = nn.Sigmoid()
    rewards = torch.log(m(dis_pred))
    cumulative_rewards = []

    rewards = torch.unbind(rewards, dim=1)
    for t in range(seq_length):
        cum_value = torch.zeros(batch_size)
        if use_cuda:
            cum_value = cum_value.cuda()
        for s in range(t, seq_length):
            discount = (gamma ** (s - t))
            cum_value += discount * rewards[s]
        cumulative_rewards.append(cum_value)
    cumulative_rewards = torch.stack(cumulative_rewards, dim=1)

    loss = nn.MSELoss()
    cumulative_rewards = cumulative_rewards.detach()
    critic_loss = loss(est_val, cumulative_rewards)

    baselines = torch.unbind(est_val, dim=1)

    ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
    advantages = []
    final_gen_objective = 0
    gen_log_prob = torch.unbind(gen_log_prob, dim=1)

    #print("Gen log prob : ")
    #print(gen_log_prob)

    #print("critic loss unbound - baselines : ")
    #print(baselines)

    for t in range(seq_length):
        log_probability = gen_log_prob[t]
        cum_advantage = torch.zeros(batch_size)
        if use_cuda:
            cum_advantage = cum_advantage.cuda()
        for s in range(t, seq_length):
            cum_advantage += (gamma ** (s - t)) * rewards[s]

        #print("Cum advantage plain for t:", t)
        #print(cum_advantage)
        cum_advantage -= baselines[t]
        #print("Cum advantage - baselines for t:", t)
        #print(cum_advantage)
        # Clip advantages.
        cum_advantage = torch.clamp(cum_advantage, -config.advantage_clipping, config.advantage_clipping)
        #print("Cum advantage - baselines after clamping for t:", t)
        #print(cum_advantage)
        advantages.append(cum_advantage)
        final_gen_objective += log_probability * cum_advantage
        #print("Final gen objective for t:", t)
        #print(final_gen_objective)

    final_gen_obj = torch.sum(final_gen_objective)/(batch_size*seq_length)
    #print(final_gen_obj)

    return final_gen_obj, critic_loss


class Discriminator(nn.Module):

    def __init__(self, encoder, decoder, use_cuda=False):
        super(Discriminator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, seq_length, batch_size):
        e_out, e_hid = self.encoder(input, batch_size)
        dis_out = decoder_train(input, e_hid, self.decoder, seq_length, batch_size, self.use_cuda)
        dis_out2 = torch.sum(dis_out, dim=1) / batch_size
        dis_sig = self.sigmoid(dis_out2)
        return dis_out, dis_sig

