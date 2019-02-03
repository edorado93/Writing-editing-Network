import configurations
from eval import Evaluate
from utils import Vectorizer, headline2abstractdataset, load_embeddings
from seq2seq.fb_seq2seq import FbSeq2seq
from seq2seq.EncoderRNN import EncoderRNN
from seq2seq.DecoderRNNFB import DecoderRNNFB
from seq2seq.ContextEncoder import ContextEncoder
import torch
import torch.nn as nn
import os
import json


class ModelManager:
    def __init__(self, args):

        self.args = args

        # Model Configuration to execute.
        self.config = configurations.init(args.dataset)[args.conf]
        if args.local_rank == 0:
            print("Config is", args.conf)

        # Model's checkpoint filename.
        v = vars(self.args)
        v['save'] = "models/" + self.config.experiment_name + '.pkl'

        # Set the random seed manually for reproducibility.
        self.seed()

        # Evaluation API for calculating the BLEU, METEOR and ROUGE scores
        self.validation_eval = Evaluate()

        # Training and Validation datasets
        self.training_abstracts, self.validation_abstracts = self.load_datasets()

        # THE model!
        self.model = self.initialize_model()

    def seed(self):
        args = self.args
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)

    def load_datasets(self):
        config = self.config
        args = self.args
        cwd = os.getcwd()
        vectorizer = Vectorizer(min_frequency=config.min_freq)

        data_path = cwd + config.relative_data_path
        training_abstracts = headline2abstractdataset(data_path, vectorizer, args.cuda, max_len=1000,
                                             use_topics=config.use_topics, use_structure_info=config.use_labels)

        validation_data_path = cwd + config.relative_dev_path
        validation_abstracts = headline2abstractdataset(validation_data_path, vectorizer, args.cuda, max_len=1000,
                                                        use_topics=config.use_topics,
                                                        use_structure_info=config.use_labels)
        if args.local_rank == 0:
            print("number of training examples: %d" % len(training_abstracts), flush=True)
        return training_abstracts, validation_abstracts

    def initialize_model(self):
        config = self.config
        args = self.args
        training_abstracts = self.training_abstracts

        context_encoder = None
        vocab_size = len(training_abstracts.vectorizer.word2idx)
        embedding = nn.Embedding(vocab_size, config.emsize, padding_idx=0)

        if config.pretrained:
            embedding = load_embeddings(embedding, training_abstracts.vectorizer.word2idx, config.pretrained, config.emsize)

        if config.use_topics or config.use_labels:
            context_encoder = ContextEncoder(config.context_dim, len(training_abstracts.vectorizer.context_vectorizer))

        title_encoder_rnn_dim = config.emsize + (config.use_topics * training_abstracts.max_context_length) * config.context_dim
        # Word embedding dim + (No. of topics * topical dim) + Structural embedding dim = 2
        abstract_encoder_rnn_dim = config.emsize + (config.use_topics * training_abstracts.max_context_length) * config.context_dim + 2

        structure_labels = {"introduction": training_abstracts.vectorizer.structural_vectorizer["introduction"],
                            "body": training_abstracts.vectorizer.structural_vectorizer["body"],
                            "conclusion": training_abstracts.vectorizer.structural_vectorizer["conclusion"],
                            "full_stop": training_abstracts.vectorizer.word2idx["."],
                            "question_mark": training_abstracts.vectorizer.word2idx["?"]}

        encoder_title = EncoderRNN(vocab_size, embedding, training_abstracts.head_len, title_encoder_rnn_dim,
                                   abstract_encoder_rnn_dim, input_dropout_p=config.input_dropout_p, output_dropout_p=config.output_dropout_p,
                                   n_layers=config.nlayers, bidirectional=config.bidirectional,
                                   rnn_cell=config.cell)
        encoder = EncoderRNN(vocab_size, embedding, training_abstracts.abs_len, abstract_encoder_rnn_dim,
                             abstract_encoder_rnn_dim, input_dropout_p=config.input_dropout_p, output_dropout_p=config.output_dropout_p,
                             variable_lengths=False, n_layers=config.nlayers,
                             bidirectional=config.bidirectional, rnn_cell=config.cell)
        decoder = DecoderRNNFB(vocab_size, embedding, training_abstracts.abs_len, abstract_encoder_rnn_dim, sos_id=2, eos_id=1,
                               n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                               input_dropout_p=config.input_dropout_p, dropout_p=config.dropout_p,
                               output_dropout_p=config.output_dropout_p, labels=structure_labels,
                               use_labels=config.use_labels, context_model=context_encoder, use_cuda=args.cuda,
                               use_intra_attention=config.use_intra_attention,
                               intra_attention_window_size=config.window_size_attention)
        model = FbSeq2seq(encoder_title, encoder, context_encoder, decoder)
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
        if args.local_rank == 0:
            print("Configuration is as follows", json.dumps({"training data path": config.relative_data_path,
                                                             "validation data path": config.relative_dev_path,
                                                             "training batch size": config.batch_size,
                                                             "word embedding dim": config.emsize,
                                                             "context embedding dim": config.context_dim,
                                                             "validation batch size": config.validation_batch_size,
                                                             "input dropout": config.input_dropout_p,
                                                             "dropout": config.dropout_p,
                                                             "output dropout": config.output_dropout_p,
                                                             "data parallel": config.data_parallel,
                                                             "distributed data parallel": config.distributed_data_parallel,
                                                             "log_interval": config.log_interval,
                                                             "print_running_loss": config.print_running_loss,
                                                             "learning rate": config.lr,
                                                             "pre-trained embeddings location": config.pretrained,
                                                             "use topics": config.use_topics,
                                                             "use labels": config.use_labels,
                                                             "use intra-attention": config.use_intra_attention,
                                                             "intra-attention window size": config.window_size_attention,
                                                             "experiment name": config.experiment_name},
                                                            sort_keys=True, indent=4, separators=(',', ': ')), flush=True)
            print('Model total parameters:', total_params, flush=True)

        return model

    def get_model(self):
        return self.model

    def get_training_data(self):
        return self.training_abstracts

    def get_validation_data(self):
        return self.validation_abstracts

    def get_config(self):
        return self.config

    def get_eval_object(self):
        return self.validation_eval