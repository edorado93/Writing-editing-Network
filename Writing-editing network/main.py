import time, argparse, math, os, sys, pickle, copy
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
from predictor import Predictor
from tensorboardX import SummaryWriter
import configurations
from pprint import pprint
import os.path
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

validation_eval = Evaluate()
cwd = os.getcwd()
vectorizer = Vectorizer(min_frequency=config.min_freq)

data_path = cwd + config.relative_data_path
abstracts = headline2abstractdataset(data_path, vectorizer, args.cuda, max_len=1000, use_topics=config.use_topics, use_structure_info=config.use_labels)

validation_data_path = cwd + config.relative_dev_path
validation_abstracts = headline2abstractdataset(validation_data_path, vectorizer, args.cuda, max_len=1000, use_topics=config.use_topics, use_structure_info=config.use_labels)

print("number of training examples: %d" % len(abstracts))

context_encoder = None
vocab_size = len(vectorizer.word2idx)
embedding = nn.Embedding(vocab_size, config.emsize, padding_idx=0)

if config.pretrained:
    embedding = load_embeddings(embedding, abstracts.vectorizer.word2idx, config.pretrained, config.emsize)

if config.use_topics or config.use_labels:
    context_encoder = ContextEncoder(config.context_dim, len(vectorizer.context_vectorizer), config.emsize)

title_encoder_rnn_dim = config.emsize + (config.use_topics * abstracts.max_context_length) * config.context_dim
abstract_encoder_rnn_dim = config.emsize + (config.use_labels + config.use_topics * abstracts.max_context_length) * config.context_dim

structure_labels = {"introduction" : abstracts.vectorizer.context_vectorizer["introduction"],
                    "body" : abstracts.vectorizer.context_vectorizer["body"],
                    "conclusion": abstracts.vectorizer.context_vectorizer["conclusion"],
                    "full_stop": abstracts.vectorizer.word2idx["."],
                    "question_mark": abstracts.vectorizer.word2idx["?"]}

encoder_title = EncoderRNN(vocab_size, embedding, abstracts.head_len, title_encoder_rnn_dim, abstract_encoder_rnn_dim, input_dropout_p=config.dropout,
                     n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
encoder = EncoderRNN(vocab_size, embedding, abstracts.abs_len, abstract_encoder_rnn_dim, abstract_encoder_rnn_dim, input_dropout_p=config.dropout, variable_lengths = False,
                  n_layers=config.nlayers, bidirectional=config.bidirectional, rnn_cell=config.cell)
decoder = DecoderRNNFB(vocab_size, embedding, abstracts.abs_len, abstract_encoder_rnn_dim, sos_id=2, eos_id=1,
                     n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, labels=structure_labels, use_labels=config.use_labels, context_model=context_encoder, use_cuda=args.cuda)
model = FbSeq2seq(encoder_title, encoder, context_encoder, decoder)
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Model total parameters:', total_params, flush=True)
configurations.print_config(config)

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

def train_batch(input_variable, input_lengths, target_variable, topics, structure_abstracts, model,
                teacher_forcing_ratio, is_eval=False):
    loss_list = []
    # Forward propagation
    prev_generated_seq = None
    target_variable_reshaped = target_variable[:, 1:].contiguous().view(-1)

    # Data structures used to store sentences for measuring BLEU scores.
    sentences = []
    drafts = [[] for _ in range(config.num_exams)]
    # Iterate over the title and the original abstract and store them as a tuple in the sentences array.
    for t in target_variable:
        sentences.append(" ".join([str(tok.item()) for tok in t if tok.item() != 0 and tok.item() != 1 and tok.item() != 2]))

    for i in range(config.num_exams):
        decoder_outputs, _, other = \
            model(input_variable, prev_generated_seq, input_lengths,
                   target_variable, teacher_forcing_ratio, topics=topics, structure_abstracts=structure_abstracts)

        decoder_outputs_reshaped = decoder_outputs.view(-1, vocab_size)
        lossi = criterion(decoder_outputs_reshaped, target_variable_reshaped)
        loss_list.append(lossi.item())
        if not is_eval:
            model.zero_grad()
            lossi.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        # Current output of the model. This will be the previously generated abstract for the model.
        prev_generated_seq = torch.squeeze(torch.topk(decoder_outputs, 1, dim=2)[1]).view(-1, decoder_outputs.size(1))

        # If we are in eval mode, obtain words for generated sequences. This will be used for the BLEU score.
        if is_eval:
            for p in prev_generated_seq:
                drafts[i].append(" ".join([str(tok.item()) for tok in p if tok.item() != 0 and tok.item() != 1 and tok.item() != 2]))
        prev_generated_seq = _mask(prev_generated_seq)

    if is_eval:
        return loss_list, sentences, drafts

    return loss_list

def bleu_scoring(abstracts, drafts):
    refs = {}
    cands = []

    for i, ab in enumerate(abstracts):
        refs[i] = [ab]

    for k in range(config.num_exams):
        cands.append({})
        for j, dr in enumerate(drafts[k]):
            cands[k][j] = dr

    # cands and refs is our input for the BLEU scoring functions.
    scores = []
    fields = ["Bleu_4", "METEOR", "ROUGE_L"]
    for i in range(3):
        scores.append([])
    for i in range(config.num_exams):
        final_scores = validation_eval.evaluate(live=True, cand=cands[i], ref=refs)
        for j in range(3):
            scores[j].append(final_scores[fields[j]])
    return scores

def evaluate(validation_dataset, model, teacher_forcing_ratio):
    validation_loader = DataLoader(validation_dataset, config.validation_batch_size)
    model.eval()
    epoch_loss_list = [0] * config.num_exams
    abstracts = []
    drafts = [[] for _ in range(config.num_exams)]
    for batch_idx, data in enumerate(validation_loader):
        topics = data[3] if config.use_topics else None
        structure_abstracts = (data[4] if config.use_topics else data[3]) if config.use_labels else None

        input_variables = data[0]
        target_variables = data[1]
        input_lengths = data[2]
        # Run the model on the validation data set WITH teacher forcing
        loss_list, batch_sentences, batch_drafts = train_batch(input_variables, input_lengths,
                                target_variables, topics, structure_abstracts, model, teacher_forcing_ratio=teacher_forcing_ratio, is_eval=True)
        num_examples = len(input_variables)
        abstracts.extend(batch_sentences)
        for i in range(config.num_exams):
            epoch_loss_list[i] += loss_list[i] * num_examples
            drafts[i].extend(batch_drafts[i])

    scores = bleu_scoring(abstracts, drafts)
    for i in range(config.num_exams):
        epoch_loss_list[i] /= float(len(validation_loader.dataset))
    return epoch_loss_list, scores

def train_epoches(start_epoch, dataset, model, n_epochs, teacher_forcing_ratio, prev_epoch_loss_list):
    train_loader = DataLoader(dataset, config.batch_size)
    patience = 0
    start = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        if time.time() - start >= 82400:
            print("Exiting with code 99", flush=True)
            exit(99)
        model.train(True)
        epoch_examples_total = 0
        total_examples = 0
        start = time.time()
        epoch_start_time = start
        total_loss = 0
        training_loss_list = [0] * config.num_exams
        for batch_idx, data in enumerate(train_loader):
            topics = data[3] if config.use_topics else None
            structure_abstracts = (data[4] if config.use_topics else data[3]) if config.use_labels else None

            input_variables = data[0]
            target_variables = data[1]
            input_lengths = data[2]
            # train model
            loss_list = train_batch(input_variables, input_lengths,
                               target_variables, topics, structure_abstracts, model, teacher_forcing_ratio)
            # Record average loss
            num_examples = len(input_variables)
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

        validation_loss, eval_scores = evaluate(validation_abstracts, model, teacher_forcing_ratio)
        if config.use_topics:
            plot_topical_encoding(vectorizer.context_vectorizer, model.context_encoder.embedding, writer, epoch)
        for i in range(config.num_exams):
            training_loss_list[i] /= float(epoch_examples_total)
            writer.add_scalar('loss/train/train_loss_abstract_'+str(i), training_loss_list[i], epoch)
            writer.add_scalar('loss/valid/validation_loss_abstract_' + str(i), validation_loss[i], epoch)
            writer.add_scalar('eval_scores/BLEU_' + str(i), eval_scores[0][i], epoch)
            writer.add_scalar('eval_scores/METEOR_' + str(i), eval_scores[1][i], epoch)
            writer.add_scalar('eval_scores/ROUGLE_' + str(i), eval_scores[2][i], epoch)

        print('****************** | end of epoch {:3d} | time: {:5.2f}s *********************'.format(epoch,  (time.time() - epoch_start_time)))
        print("Validation Loss: ")
        pprint(validation_loss)
        print("BLEU-4:")
        pprint(eval_scores[0])
        print("METEOR:")
        pprint(eval_scores[1])
        print("ROUGLE-L:")
        pprint(eval_scores[2])

        # Use BLEU score as yardstick for early stopping rather than the validation loss.
        if prev_epoch_loss_list[1] > eval_scores[0][1]:
            patience += 1
            if patience == config.patience:
                print("Breaking off now. Performance has not improved on validation set since the last",config.patience,"epochs")
                break
        else:
            patience = 0
            prev_epoch_loss_list = eval_scores[0][:]
            save_model(epoch, prev_epoch_loss_list)
            print("Saved best model till now!")

def save_model(epoch, prev_epoch_loss_list):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_eval_scores': prev_epoch_loss_list}
    torch.save(state, args.save)

def load_checkpoint():
    start_epoch = 0
    prev_epoch_loss_list = [0.] * config.num_exams
    if os.path.isfile(args.save):
        print("=> loading checkpoint '{}'".format(args.save))
        checkpoint = torch.load(args.save)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Only for older models trained without this
        if 'best_eval_scores' in checkpoint:
            prev_epoch_loss_list = checkpoint['best_eval_scores']
        print("=> loaded checkpoint '{}' (epoch {}) best_eval_scores {}"
              .format(args.save, checkpoint['epoch'], prev_epoch_loss_list))
    else:
        print("=> no checkpoint found at '{}', starting from scratch".format(args.save))

    return start_epoch, prev_epoch_loss_list

if __name__ == "__main__":
    if args.mode == 0:
        # train
        try:
            start_epoch, prev_epoch_loss_list = load_checkpoint()
            train_epoches(start_epoch, abstracts, model, config.epochs, teacher_forcing_ratio=1, prev_epoch_loss_list=prev_epoch_loss_list)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    elif args.mode == 1:
        load_checkpoint()
        predictor = Predictor(model, abstracts.vectorizer, use_cuda=args.cuda)
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
            outputs = predictor.predict(seq, num_exams, max_length=max_length, topics=topics, use_structure=config.use_labels)
            for i in range(num_exams):
                print(i)
                print(outputs[i])
            print('-' * 120)
            count += 1
    elif args.mode == 2:
        num_exams = 3
        # predict file
        load_checkpoint()
        predictor = Predictor(model, abstracts.vectorizer, use_cuda=args.cuda)
        print("number of test examples: %d" % len(validation_abstracts))
        f_out_name = cwd + config.relative_gen_path
        outputs = []
        title = []
        for j in range(num_exams):
            outputs.append([])
        i = 0
        print("Start generating:")
        train_loader = DataLoader(validation_abstracts, config.batch_size)
        for batch_idx, (source, target, input_lengths) in enumerate(train_loader):
            output_seq = predictor.predict_batch(source, input_lengths.tolist(), num_exams)
            for seq in output_seq:
                title.append(seq[0])
                for j in range(num_exams):
                    outputs[j].append(seq[j+1])
                i += 1
                if i % 100 == 0:
                    print("Percentages:  %.4f" % (i/float(len(validation_abstracts))))

        print("Start writing:")
        for i in range(num_exams):
            out_name = f_out_name % i
            f_out = open(out_name, 'w')
            for j in range(len(title)):
                f_out.write(title[j] + '\n' + outputs[i][j] + '\n\n')
                if j % 100 == 0:
                    print("Percentages:  %.4f" % (j/float(len(validation_abstracts))))
            f_out.close()
        f_out.close()
    elif args.mode == 3:
        test_data_path = cwd + config.relative_test_path
        test_abstracts = headline2abstractdataset(test_data_path, vectorizer, args.cuda, max_len=1000,
                                                        use_topics=config.use_topics,
                                                        use_structure_info=config.use_labels)
        load_checkpoint()
        test_loader = DataLoader(test_abstracts, config.batch_size)
        eval_f = Evaluate()
        num_exams = 3
        predictor = Predictor(model, validation_abstracts.vectorizer, use_cuda=args.cuda)
        print("Start Evaluating")
        print("Test Data: ", len(validation_abstracts))
        cand, ref = predictor.preeval_batch(test_loader, len(validation_abstracts), num_exams, use_topics=config.use_topics, use_labels=config.use_labels)
        scores = []
        fields = ["Bleu_4", "METEOR", "ROUGE_L"]
        for i in range(3):
            scores.append([])
        for i in range(num_exams):
            print("No.", i)
            final_scores = eval_f.evaluate(live=True, cand=cand[i], ref=ref)
            for j in range(3):
                scores[j].append(final_scores[fields[j]])
        pprint(scores)
        with open('figure.pkl', 'wb') as f:
            pickle.dump((fields, scores), f)
