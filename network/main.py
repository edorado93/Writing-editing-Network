import time, argparse, os, sys, pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn
from utils import headline2abstractdataset
from predictor import Predictor
from seq2seq.model_manager import ModelManager
from seq2seq.stat_manager import StatManager, AverageMeter
from seq2seq.distributed_sequential_sampler import DistributedSequentialSampler
from pprint import pprint
import os.path
sys.path.insert(0,'..')
from eval import Evaluate

manager, model, criterion, optimizer, train_sampler, stat_manager = None, None, None, None, None, None

def make_parser():
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

    return parser.parse_known_args()

def init(args):
    cudnn.benchmark = True
    manager = ModelManager(args)
    train_sampler = None
    model = manager.get_model()
    config = manager.get_config()
    stat_manager = StatManager(config, is_testing=False)
    training_abstracts = manager.get_training_data()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        model = model.cuda()
        criterion = criterion.cuda()

    if config.dataparallel and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        train_sampler = DistributedSequentialSampler(training_abstracts)
        config.batch_size //= torch.cuda.device_count()

    return manager, model, criterion, optimizer, train_sampler, stat_manager

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
    vocab_size = len(training_abstracts.vectorizer.word2idx)
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
        final_scores = manager.get_eval_object().evaluate(live=True, cand=cands[i], ref=refs)
        for j in range(3):
            scores[j].append(final_scores[fields[j]])
    return scores

def evaluate(validation_dataset, model, teacher_forcing_ratio):
    validation_loader = DataLoader(validation_dataset, config.validation_batch_size)
    model.eval()
    abstracts = []
    validation_loss = AverageMeter()
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

        # Updayte average validation loss
        validation_loss.update(loss_list[1], input_variables.size(0))

        abstracts.extend(batch_sentences)
        for i in range(config.num_exams):
            drafts[i].extend(batch_drafts[i])

    scores = bleu_scoring(abstracts, drafts)
    return validation_loss, scores

def train_epoches(start_epoch, dataset, model, n_epochs, teacher_forcing_ratio, prev_epoch_loss_list):
    train_loader = DataLoader(dataset, config.batch_size, sampler=train_sampler)
    patience = 0
    cutoff_time_start = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        if time.time() - cutoff_time_start >= 82400:
            print("Exiting with code 99", flush=True)
            exit(99)

        training_loss = AverageMeter()
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
            loss_list = train_batch(input_variables, input_lengths,
                               target_variables, topics, structure_abstracts, model, teacher_forcing_ratio)

            # Update the training loss.
            training_loss.update(loss_list[1], input_variables.size(0))

            # Update batch processing times.
            batch_time.update(time.time() - start)
            start = time.time()

            # Log after a fixed interval
            if batch_idx % config.log_interval == 0:
                print('PID_{} | epoch {:3d} | {:5d}/{:5d} examples | lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.2f}'.format(args.local_rank, epoch, batch_idx * config.batch_size, len(train_sampler) if train_sampler else len(train_loader.dataset), optimizer.param_groups[0]['lr'],
                                           batch_time.avg * 1000, training_loss.avg), flush=True)

        if args.local_rank == 0:
            validation_loss, eval_scores = evaluate(validation_abstracts, model, teacher_forcing_ratio)

            stat_manager.log_original_training_loss(training_loss.avg, epoch)
            stat_manager.log_original_validation_loss(validation_loss.avg, epoch)


            print('PID_{} ****************** | end of epoch {:3d} | time: {:5.2f}s *********************'.format(args.local_rank, epoch,  (time.time() - epoch_start_time)), flush=True)
            print("PID_{} Validation Loss: {}, BLEU-4: {}, METEOR: {}, ROUGLE-L: {}".format(args.local_rank, validation_loss.avg, eval_scores[0][1],
                                                                                     eval_scores[1][1], eval_scores[2][1]), flush=True)

            # Use BLEU score as yardstick for early stopping rather than the validation loss.
            if prev_epoch_loss_list[1] > eval_scores[0][1]:
                patience += 1
                if patience == config.patience:
                    print("Breaking off now. Performance has not improved on validation set since the last",config.patience,"epochs", flush=True)
                    exit(0)
            else:
                patience = 0
                prev_epoch_loss_list = eval_scores[0][:]
                save_model(epoch, prev_epoch_loss_list)
                print("Saved best model till now!", flush=True)
            print("", flush=True)

def save_model(epoch, prev_epoch_loss_list):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_eval_scores': prev_epoch_loss_list}
    torch.save(state, args.save)

def load_checkpoint():
    start_epoch = 0
    prev_epoch_loss_list = [0.] * config.num_exams
    if os.path.isfile(args.save):
        print("=> loading checkpoint '{}'".format(args.save), flush=True)
        checkpoint = torch.load(args.save)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Only for older models trained without this
        if 'best_eval_scores' in checkpoint:
            prev_epoch_loss_list = checkpoint['best_eval_scores']
        print("=> loaded checkpoint '{}' (epoch {}) best_eval_scores {}"
              .format(args.save, checkpoint['epoch'], prev_epoch_loss_list), flush=True)
    else:
        print("=> no checkpoint found at '{}', starting from scratch".format(args.save), flush=True)

    return start_epoch, prev_epoch_loss_list

if __name__ == "__main__":
    args, unknown = make_parser()
    manager, model, criterion, optimizer, train_sampler, stat_manager = init(args)
    config = manager.get_config()
    validation_abstracts = manager.get_validation_data()
    training_abstracts = manager.get_training_data()
    v = vars(args)
    v['save'] = "models/"+config.experiment_name + '.pkl'

    if args.mode == 0:
        # train
        try:
            start_epoch, prev_epoch_loss_list = load_checkpoint()
            train_epoches(start_epoch, training_abstracts, model, config.epochs, teacher_forcing_ratio=1, prev_epoch_loss_list=prev_epoch_loss_list)
        except KeyboardInterrupt:
            print('-' * 89, flush=True)
            print('Exiting from training early', flush=True)
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
