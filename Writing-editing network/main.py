import time, argparse, math, os, sys, pickle, copy, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.backends import cudnn
from utils import Vectorizer, headline2abstractdataset, load_embeddings, plot_topical_encoding
from seq2seq.fb_seq2seq import FbSeq2seq
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNNFB import DecoderRNNFB
from seq2seq.ContextEncoder import ContextEncoder
from seq2seq.discriminator import reinforce, Encoder, DecoderRNN, Discriminator, Critic
from predictor import Predictor
from tensorboardX import SummaryWriter
import configurations
from pprint import pprint
sys.path.insert(0,'..')
from eval import Evaluate

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='seq2seq model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_sentence(1)/predict_file(2) or evaluate(3)')
parser.add_argument('--conf', type=str,
                    help="configuration to load for the training")
args = parser.parse_args()
config = configurations.get_conf(args.conf)
writer = SummaryWriter("saved_runs/" + config.experiment_name)
v = vars(args)
v['save'] = "models/"+config.experiment_name + '.pkl'

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

cwd = os.getcwd()
vectorizer = Vectorizer(min_frequency=config.min_freq)
train_shuffle_samples = []

validation_data_path = cwd + config.relative_dev_path
validation_abstracts = headline2abstractdataset(validation_data_path, vectorizer, args.cuda, max_len=1000)

data_path = cwd + config.relative_data_path
abstracts = headline2abstractdataset(data_path, vectorizer, args.cuda, max_len=1000)
print("number of training examples: %d" % len(abstracts))

vocab_size = abstracts.vectorizer.vocabulary_size
embedding = nn.Embedding(vocab_size, config.emsize, padding_idx=0)

if config.pretrained:
    embedding = load_embeddings(embedding, abstracts.vectorizer.word2idx, config.pretrained, config.emsize)

if config.use_topics:
    context_encoder = ContextEncoder(config.context_dim, len(vectorizer.context_vectorizer), config.emsize)
    max_topics = abstracts.max_context_length
    new_embedding_size = max_topics * config.context_dim + config.emsize
else:
    context_encoder = None
    new_embedding_size = config.emsize

encoder_title = EncoderRNN(vocab_size, embedding, abstracts.head_len, new_embedding_size, input_dropout_p=config.dropout,
                     n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
encoder = EncoderRNN(vocab_size, embedding, abstracts.abs_len, new_embedding_size, input_dropout_p=config.dropout, variable_lengths = False,
                  n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
decoder = DecoderRNNFB(vocab_size, embedding, abstracts.abs_len, new_embedding_size, sos_id=2, eos_id=1,
                     n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout)
model = FbSeq2seq(encoder_title, encoder, context_encoder, decoder)

""" Define the Discriminator model here """

discrim_encoder = Encoder(config.emsize, config.emsize, vocab_size, config.batch_size)
discrim_decoder = DecoderRNN(config.emsize, config.emsize, vocab_size, 1, config.batch_size)
discrim_model = Discriminator(discrim_encoder, discrim_decoder)
discrim_criterion = nn.BCELoss()
critic_model = Critic(config.emsize, config.emsize, vocab_size)

""" Ends here """


total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Model total parameters:', total_params, flush=True)

if config.dataparallel and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=0)
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Mask variable
def _mask(prev_generated_seq):
    prev_mask = torch.eq(prev_generated_seq, 1)
    lengths = torch.argmax(prev_mask,dim=1)
    max_len = prev_generated_seq.size(1)
    mask = []
    for i in range(prev_generated_seq.size(0)):
        if lengths[i] == 0:
            mask_line = [0] * max_len
        else:
            mask_line = [0] * lengths[i].item()
            mask_line.extend([1] * (max_len - lengths[i].item()))
        mask.append(mask_line)
    mask = torch.ByteTensor(mask)
    if args.cuda:
        mask = mask.cuda()
    return prev_generated_seq.data.masked_fill_(mask, 0)

def load_training_samples_for_shuffling(dataset):
    train_loader = DataLoader(dataset, config.batch_size)
    for d in train_loader:
        train_shuffle_samples.append(d)

def freeze_generator():
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_generator():
    for param in model.parameters():
        param.requires_grad = True

def freeze_discriminator():
    for param in discrim_model.parameters():
        param.requires_grad = False

def unfreeze_discriminator():
    for param in discrim_model.parameters():
        param.requires_grad = True


def train_discriminator(input_variable, target_variable, is_eval=False):
    loss_list = []
    '''add other return values'''
    output = discrim_model(input_variable)
    lossi = discrim_criterion(output, target_variable)
    loss_list.append(lossi)
    """ Check if we need this if condition here, since we are freezing the weights anyhow """
    if not is_eval:
        discrim_model.zero_grad()
        lossi.backward(retain_graph=True)
        optimizer.step()
    '''Need to add code to train the critic when we train the discriminator'''

def train_generator(input_variable, input_lengths, target_variable, topics, model,
                teacher_forcing_ratio, is_eval=False):
    loss_list = []
    # Forward propagation
    prev_generated_seq = None
    target_variable_reshaped = target_variable[:, 1:].contiguous().view(-1)

    sentences = []
    drafts = [[] for _ in range(config.num_exams)]
    probabilities = [[] for _ in range(config.num_exams)]
    for i, t in zip(input_variable, target_variable):
        sentences.append((" ".join([vectorizer.idx2word[tok.item()] for tok in i if tok.item() != 0 and tok.item() != 1 and tok.item() != 2]),
                          " ".join([vectorizer.idx2word[tok.item()] for tok in t if tok.item() != 0 and tok.item() != 1 and tok.item() != 2])))

    for i in range(config.num_exams):
        topics = topics if config.use_topics else None
        decoder_outputs, _, other = \
            model(input_variable, prev_generated_seq, input_lengths,
                   target_variable, teacher_forcing_ratio, topics)

        decoder_outputs_reshaped = decoder_outputs.view(-1, vocab_size)
        prev_generated_seq = torch.squeeze(torch.topk(decoder_outputs, 1, dim=2)[1]).view(-1, decoder_outputs.size(1))
        prev_generated_seq = _mask(prev_generated_seq)

        log_probabilities = torch.squeeze(torch.topk(decoder_outputs, 1, dim=2)[0]).view(-1, decoder_outputs.size(1))
        for lp_tensor, p_tensor in zip(log_probabilities, prev_generated_seq):
            drafts[i].append(p_tensor)
            probabilities[i].append(lp_tensor)

        # Only calculate the reinforce loss the generator is being trained i.e.
        # this is not the eval mode.
        if not is_eval:
            """ Call Discriminator, Critic and get the ReINFORCE Loss Term"""
            #input is the batch_size * sequence length of woord to index of abstracts
            est_values = critic_model(input)
            dis_out = discrim_model(input)
            #gen_log is the log probabilities of generator output
            reinforce_loss = reinforce(gen_log, dis_out, est_values, seq_length, config)
        else:
            reinforce_loss = 0

        lossi = criterion(decoder_outputs_reshaped, target_variable_reshaped) + reinforce_loss
        loss_list.append(lossi.item())
        if not is_eval:
            model.zero_grad()
            lossi.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

    return loss_list, sentences, drafts

def test_code(dataset):
    load_training_samples_for_shuffling(dataset)
    source, target, input_lengths, topics = random.choice(train_shuffle_samples)
    train_batch(source, input_lengths, target, topics, 1, False)

def train_batch(input_variables, input_lengths, target_variables, topics, teacher_forcing_ratio, is_generator):
    if is_generator:
        unfreeze_generator()
        freeze_discriminator()
        loss_list, sentences, drafts = train_generator(input_variables, input_lengths, target_variables, topics, model, teacher_forcing_ratio)
        return loss_list
    else:
        # unfreeze_discriminator()
        freeze_generator()
        loss_list, sentences, drafts = train_generator(input_variables, input_lengths, target_variables,
                                                                      topics, model, teacher_forcing_ratio, is_eval=True)
        # Randomly select a batch of data from the actual training set.
        _, true_target, _, _ = random.choice(train_shuffle_samples)
        discriminator_input_variables = []
        discriminator_target_variables = []

        # Batch size is the same, so we can zip
        for d, t in zip(drafts[1], true_target):

            eos_tensor = torch.tensor(1).cuda() if input_variables.is_cuda else torch.tensor(1)
            d =  torch.cat((d.view(1, -1), eos_tensor.view(1, -1)), dim=1)
            t = t.view(1, -1)

            # Half the number of samples should be from the actual dataset and remaining half from genera
            # -ted samples.
            if random.random() > 0.5:
                discriminator_input_variables.append(t)
                # CONFIRM ? The discriminator should output 1 if the abstract is from the true data
                discriminator_target_variables.append(1.)
            else:
                discriminator_input_variables.append(d)
                # CONFIRM ? The discriminator should output 0 if the abstract is from the generated data
                discriminator_target_variables.append(0.)

        discriminator_input_variables = torch.stack(discriminator_input_variables)
        discriminator_target_variables = torch.tensor(discriminator_target_variables).cuda() if input_variables.is_cuda else torch.tensor(discriminator_target_variables)
        print(discriminator_input_variables.shape, discriminator_target_variables.shape)
        if input_variables.is_cuda:
            discriminator_input_variables = discriminator_input_variables.cuda()
            discriminator_target_variables =discriminator_target_variables.cuda()

        print(discriminator_input_variables.shape, discriminator_target_variables.shape)
        """ Mix them with the true data and pass it to the discriminator """
        # output = train_discriminator(discriminator_input_variables, discriminator_target_variables)
        # return output


def evaluate(validation_dataset, model, teacher_forcing_ratio):
    validation_loader = DataLoader(validation_dataset, config.batch_size)
    model.eval()
    epoch_loss_list = [0] * config.num_exams
    for batch_idx, (source, target, input_lengths, topics) in enumerate(validation_loader):
        input_variables = source
        target_variables = target
        # train model
        loss_list, _, _ = train_generator(input_variables, input_lengths, target_variables,
                                                              topics, model, teacher_forcing_ratio, is_eval=True)
        num_examples = len(source)
        for i in range(config.num_exams):
            epoch_loss_list[i] += loss_list[i] * num_examples
    for i in range(config.num_exams):
        epoch_loss_list[i] /= float(len(validation_loader.dataset))
    return epoch_loss_list

def train_epoches(dataset, model, n_epochs, teacher_forcing_ratio):
    train_loader = DataLoader(dataset, config.batch_size)
    prev_epoch_loss_list = [100] * config.num_exams
    patience = 0
    best_model = None
    # Loads the entire training set into memory. So that we can fetch a random batch to feed to the
    # discriminator while training.
    load_training_samples_for_shuffling(dataset)

    for epoch in range(1, n_epochs + 1):
        test_code(dataset)
        exit(0)
        model.train(True)
        epoch_examples_total = 0
        total_examples = 0
        start = time.time()
        epoch_start_time = start
        total_loss = 0
        training_loss_list = [0] * config.num_exams
        for batch_idx, (source, target, input_lengths, topics) in enumerate(train_loader):
            input_variables = source
            target_variables = target
            # train model

            # Train the GENERATOR
            loss_list = train_batch(input_variables, input_lengths,
                               target_variables, topics, teacher_forcing_ratio, True)

            # Train the DISCRIMINATOR
            train_batch(input_variables, input_lengths,
                        target_variables, topics, teacher_forcing_ratio, False)

            # Record average loss
            num_examples = len(source)
            epoch_examples_total += num_examples
            for i in range(config.num_exams):
                training_loss_list[i] += loss_list[i] * num_examples

            # Add to local variable for logging
            total_loss += loss_list[-1] * num_examples
            total_examples += num_examples
            if total_examples % config.log_interval == 0:
                cur_loss = total_loss / float(config.log_interval)
                end_time = time.time()
                elapsed = end_time - start
                start = end_time
                total_loss = 0
                print('| epoch {:3d} | {:5d}/{:5d} examples | lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.2f}'.format(
                    epoch, total_examples, len(train_loader.dataset), optimizer.param_groups[0]['lr'],
                                           elapsed * 1000 / config.log_interval, cur_loss),
                    flush=True)

        validation_loss = evaluate(validation_abstracts, model, teacher_forcing_ratio)
        if config.use_topics:
            plot_topical_encoding(vectorizer.context_vectorizer, model.context_encoder.embedding, writer, epoch)
        for i in range(config.num_exams):
            training_loss_list[i] /= float(epoch_examples_total)
            writer.add_scalar('loss/train/train_loss_abstract_'+str(i), training_loss_list[i], epoch)
            writer.add_scalar('loss/valid/validation_loss_abstract_' + str(i), validation_loss[i], epoch)

        print('| end of epoch {:3d} | valid loss {:5.2f},{:5.2f},{:5.2f} | time: {:5.2f}s'.format(epoch, validation_loss[0], validation_loss[1], validation_loss[2],
                                                                                   (time.time() - epoch_start_time)),
              flush=True)
        if prev_epoch_loss_list[:-1] < validation_loss[:-1]:
            patience += 1
            if patience == config.patience:
                print("Breaking off now. Performance has not improved on validation set since the last",config.patience,"epochs")
                break
        else:
            print("Saved best model till now!")
            best_model = copy.deepcopy(model)
            patience = 0
            prev_epoch_loss_list = validation_loss[:]
    return best_model


if __name__ == "__main__":
    if args.mode == 0:
        # train
        try:
            print("start training...")
            model = train_epoches(abstracts, model, config.epochs, teacher_forcing_ratio=1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        torch.save(model.state_dict(), args.save)
        print("model saved")
    elif args.mode == 1:
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        predictor = Predictor(model, abstracts.vectorizer)
        count = 1
        while True:
            seq_str = input("Type in a source sequence:\n")
            topics_required = input("Do you want to provide topics?").lower().strip()
            topics = None
            if topics_required in ["y", "yes"]:
                topics = input("Provide a list of comma separated topics:\n").split(',')
                topics = vectorizer.topics_to_index_tensor(topics)
            seq = seq_str.strip().split(' ')
            num_exams = int(input("Type the number of drafts:\n"))
            max_length = int(input("Type the number of words to be generated\n"))
            print("\nresult:")
            outputs = predictor.predict(seq, num_exams, max_length=max_length, topics=topics)
            for i in range(num_exams):
                print(i)
                print(outputs[i])
            print('-' * 120)
            count += 1
    elif args.mode == 2:
        num_exams = 3
        # predict file
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        predictor = Predictor(model, abstracts.vectorizer)
        data_path = cwd + config.relative_dev_path
        abstracts = headline2abstractdataset(data_path, abstracts.vectorizer, args.cuda, max_len=1000)
        print("number of test examples: %d" % len(abstracts))
        f_out_name = cwd + config.relative_gen_path
        outputs = []
        title = []
        for j in range(num_exams):
            outputs.append([])
        i = 0
        print("Start generating:")
        train_loader = DataLoader(abstracts, config.batch_size)
        for batch_idx, (source, target, input_lengths) in enumerate(train_loader):
            output_seq = predictor.predict_batch(source, input_lengths.tolist(), num_exams)
            for seq in output_seq:
                title.append(seq[0])
                for j in range(num_exams):
                    outputs[j].append(seq[j+1])
                i += 1
                if i % 100 == 0:
                    print("Percentages:  %.4f" % (i/float(len(abstracts))))

        print("Start writing:")
        for i in range(num_exams):
            out_name = f_out_name % i
            f_out = open(out_name, 'w')
            for j in range(len(title)):
                f_out.write(title[j] + '\n' + outputs[i][j] + '\n\n')
                if j % 100 == 0:
                    print("Percentages:  %.4f" % (j/float(len(abstracts))))
            f_out.close()
        f_out.close()
    elif args.mode == 3:
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        dev_data_path = cwd + config.relative_dev_path
        abstracts = headline2abstractdataset(dev_data_path, abstracts.vectorizer, args.cuda, max_len=1000)
        test_loader = DataLoader(abstracts, config.batch_size)
        eval_f = Evaluate()
        num_exams = 8
        predictor = Predictor(model, abstracts.vectorizer)
        print("Start Evaluating")
        print("Test Data: ", len(abstracts))
        cand, ref = predictor.preeval_batch(test_loader, len(abstracts), num_exams)
        scores = []
        fields = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
        for i in range(6):
            scores.append([])
        for i in range(num_exams):
            print("No.", i)
            final_scores = eval_f.evaluate(live=True, cand=cand[i], ref=ref)
            for j in range(6):
                scores[j].append(final_scores[fields[j]])
        with open('figure.pkl', 'wb') as f:
            pickle.dump((fields, scores), f)
    elif args.mode == 4:
        # predict sentence
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        # train
        try:
            print("Resume training...")
            model = train_epoches(abstracts, model, config.epochs, teacher_forcing_ratio=1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        torch.save(model.state_dict(), args.save)
        print("model saved")
