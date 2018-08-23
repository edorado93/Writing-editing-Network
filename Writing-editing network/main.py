import time, argparse, os, sys, pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.backends import cudnn
from utils import Vectorizer, headline2abstractdataset, load_embeddings
from predictor import Predictor
from seq2seq.model_manager import ModelManager
from seq2seq.stat_manager import StatManager, AverageMeter
from seq2seq.distributed_sequential_sampler import DistributedSequentialSampler
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
parser.add_argument("--local_rank", type=int, default=0,
                    help="The local rank of the process provided by the distributed launch utility, if being used")
args = parser.parse_args()

manager = ModelManager(args)
config = manager.get_config()
model = manager.get_model()
validation_abstracts = manager.get_validation_data()
training_abstracts = manager.get_training_data()
train_sampler = None

stat_manager = StatManager(config, is_testing=False)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)
if args.cuda:
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    criterion = criterion.cuda()

if config.dataparallel and torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    train_sampler = DistributedSequentialSampler(training_abstracts)
    config.batch_size //= torch.cuda.device_count()

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

def count_repetitions(generated_sequence, K):
    eos_id = 1
    number_of_words = generated_sequence.shape[1]
    batch_size = generated_sequence.shape[0]
    repetitions_in_K_window = 0
    total_words_in_window = 0
    for j in range(batch_size):
        for i in range(K, number_of_words):
            if generated_sequence[j][i].item() == eos_id:
                break
            repetitions_in_K_window += torch.sum(generated_sequence[j, i].view(-1,1) == generated_sequence[j, i - K : i]).item()
            total_words_in_window += K
    return repetitions_in_K_window, total_words_in_window

def train_batch(input_variable, input_lengths, target_variable, topics, structure_abstracts, model,
                teacher_forcing_ratio, is_eval=False):
    loss_list = [[0 for _ in range(config.num_exams)] for _ in range(3)]
    vocab_size = len(training_abstracts.vectorizer.word2idx)
    prev_generated_seq = None

    number_of_words = target_variable.shape[1]
    slice_size = 100
    number_of_slices = number_of_words / slice_size + (number_of_words % slice_size != 0)
    repetitions = 0
    total_words = 0

    # Data structures used to store sentences for measuring BLEU scores.
    sentences = []
    drafts = [[] for _ in range(config.num_exams)]
    # Iterate over the title and the original abstract and store them as a tuple in the sentences array.
    for t in target_variable:
        sentences.append(" ".join([str(tok.item()) for tok in t if tok.item() != 0 and tok.item() != 1 and tok.item() != 2]))

    for i in range(config.num_exams):
        sequence = None
        model.zero_grad()
        decoder_outputs, _, other = \
            model(input_variable, prev_generated_seq, input_lengths,
                  target_variable, teacher_forcing_ratio, topics=topics, structure_abstracts=structure_abstracts)

        new_target_variable = target_variable[:, 1:]
        for j in range(0, number_of_words, slice_size):
            target_var_slice_reshaped = new_target_variable[:, j : j + slice_size].contiguous().view(-1)
            decoder_outputs_slice = decoder_outputs[:, j : j + slice_size, :].contiguous()
            decoder_outputs_reshaped = decoder_outputs_slice.view(-1, vocab_size)
            lossi = criterion(decoder_outputs_reshaped, target_var_slice_reshaped)

            # Current output of the model. This will be the previously generated abstract for the model.
            sliced_sequence = torch.squeeze(torch.topk(decoder_outputs_slice, 1, dim=2)[1]).view(-1, decoder_outputs_slice.size(1))
            sequence = torch.cat((sequence, sliced_sequence), dim=1) if sequence is not None else sliced_sequence

            if i  == 1:
                rep, tot_words = count_repetitions(sliced_sequence, config.K)
                repetitions += rep
                total_words += tot_words

            # Cross Entropy with previous words error computation.
            detached_generated_sequence = sliced_sequence.detach()
            prev_words_CE_loss = 0
            number_of_words_in_slice = decoder_outputs_slice.shape[1]
            #TODO: Second slice onwards this should start from 0 and not K
            for k in range(config.K, number_of_words_in_slice):
                prev_words_CE_loss += cross_entropy_with_previously_generated_words(k, detached_generated_sequence,
                                                                                    decoder_outputs_slice[:, k, :], config.K, vocab_size)

            # We want to maximise the cross entropy of each word with the previous words being considered as ground truth. Hence, we subtract.
            ce_loss = prev_words_CE_loss / (number_of_words_in_slice - config.K)
            new_loss = lossi - config.cross_entropy_weight * ce_loss

            # Track all 3 losses for plotting
            loss_list[0][i] += lossi.item()
            loss_list[1][i] += ce_loss.item()
            loss_list[2][i] += new_loss.item()

            if not is_eval:
                new_loss.backward(retain_graph=True)

        for l in range(3):
            loss_list[l][i] /= number_of_slices

        if not is_eval:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        # If we are in eval mode, obtain words for generated sequences. This will be used for the BLEU score.
        if is_eval:
            for p in sequence:
                drafts[i].append(" ".join([str(tok.item()) for tok in p if tok.item() != 0 and tok.item() != 1 and tok.item() != 2]))

        prev_generated_seq = _mask(sequence)

    if is_eval:
        return loss_list, sentences, drafts

    return loss_list, repetitions, total_words

def cross_entropy_with_previously_generated_words(current_index, detached_generated_words, softmax_distribution, K, vocab_size):
    softmax_distribution = softmax_distribution.unsqueeze(1).expand(softmax_distribution.shape[0], K, vocab_size).reshape(-1, vocab_size)
    target = detached_generated_words[:, current_index - K  :current_index].contiguous().view(-1)
    return criterion(softmax_distribution, target)


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
        final_scores = manager.get_eval_object().evaluate(live=True, cand=cands[i], ref=refs)
        for j in range(3):
            scores[j].append(final_scores[fields[j]])
    return scores

def evaluate(validation_dataset, model, teacher_forcing_ratio):
    validation_loader = DataLoader(validation_dataset, config.validation_batch_size)
    model.eval()
    abstracts = []
    validation_loss = [AverageMeter() for _ in range(3)]
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

        # Update average validation loss
        for l in range(3):
            validation_loss[l].update(loss_list[l][1], input_variables.size(0))

        abstracts.extend(batch_sentences)
        for i in range(config.num_exams):
            drafts[i].extend(batch_drafts[i])

    scores = bleu_scoring(abstracts, drafts)
    return validation_loss, scores

def train_epoches(start_epoch, dataset, model, n_epochs, teacher_forcing_ratio, prev_epoch_loss_list):
    train_loader = DataLoader(dataset, config.batch_size, sampler=train_sampler)
    patience = 0
    total_repetitions = 0
    total_generated_words = 0
    cutoff_time_start = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        if time.time() - cutoff_time_start >= 82400:
            print("Exiting with code 99", flush=True)
            exit(99)

        training_loss = [AverageMeter() for _ in range(3)]
        batch_time = AverageMeter()
        epoch_start_time = time.time()

        # Put the model in training mode.
        model.train(True)

        start = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Use 1 based indexing for logging purposes
            batch_idx += 1
            topics = data[3] if config.use_topics else None
            structure_abstracts = (data[4] if config.use_topics else data[3]) if config.use_labels else None

            input_variables = data[0]
            target_variables = data[1]
            input_lengths = data[2]

            # train model
            loss_list, repetitions, total_words = train_batch(input_variables, input_lengths,
                               target_variables, topics, structure_abstracts, model, teacher_forcing_ratio)

            # Count the number of repetions. Use this to see the effect of diversity promoting objective loss.
            total_repetitions += repetitions
            total_generated_words += total_words
            # Update the training loss.
            for l in range(3):
                training_loss[l].update(loss_list[l][1], input_variables.size(0))

            # Update batch processing times.
            batch_time.update(time.time() - start)
            start = time.time()

            # Log after a fixed interval
            if batch_idx % config.log_interval == 0:
                print('PID_{} | epoch {:3d} | {:5d}/{:5d} examples | lr {:02.4f} | ms/batch {:5.2f} | '
                      'Original Loss {:5.2f}, CE Loss {:5.2f}, Subtracted Loss {:5.2f}'.format(args.local_rank, epoch, batch_idx * config.batch_size, len(train_sampler) if train_sampler else len(train_loader.dataset), optimizer.param_groups[0]['lr'],
                                           batch_time.avg * 1000, training_loss[0].avg, training_loss[1].avg, training_loss[2].avg), flush=True)

        if args.local_rank == 0:
            validation_loss, eval_scores = evaluate(validation_abstracts, model, teacher_forcing_ratio)

            stat_manager.log_original_training_loss(training_loss[0].avg, epoch)
            stat_manager.log_original_validation_loss(validation_loss[0].avg, epoch)
            stat_manager.log_cross_entropy_training_loss(training_loss[1].avg, epoch)
            stat_manager.log_cross_entropy_validation_loss(training_loss[1].avg, epoch)
            stat_manager.log_subtracted_training_loss(training_loss[2].avg, epoch)
            stat_manager.log_subtracted_validation_loss(training_loss[2].avg, epoch)


            print('PID_{} ****************** | end of epoch {:3d} | time: {:5.2f}s *********************'.format(args.local_rank, epoch,  (time.time() - epoch_start_time)))
            print("PID_{} Original Loss {:5.2f}, CE Loss {:5.2f}, Subtracted Loss {:5.2f}, BLEU-4: {}, METEOR: {}, ROUGLE-L: {}".format(args.local_rank, validation_loss[0].avg, validation_loss[1].avg, validation_loss[2].avg, eval_scores[0][1],
                                                                                     eval_scores[1][1], eval_scores[2][1]))

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
            print("")

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
            train_epoches(start_epoch, training_abstracts, model, config.epochs, teacher_forcing_ratio=1, prev_epoch_loss_list=prev_epoch_loss_list)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    elif args.mode == 1:
        load_checkpoint()
        predictor = Predictor(model, training_abstracts.vectorizer, use_cuda=args.cuda)
        count = 1
        while True:
            seq_str = input("Type in a source sequence:\n")
            topics_required = input("Do you want to provide topics?").lower().strip()
            topics = None
            if topics_required in ["y", "yes"]:
                topics = input("Provide a list of comma separated topics:\n").split(',')
                topics = training_abstracts.vectorizer.topics_to_index_tensor(topics)
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
    elif args.mode == 3:
        cwd = os.getcwd()
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
