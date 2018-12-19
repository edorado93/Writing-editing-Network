import torch
import numpy as np
import json


class Predictor(object):
    def __init__(self, model, vectorizer, use_cuda=False):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.vectorizer = vectorizer

    def predict(self, src_seq, num_exams, topics=None, max_length=None, use_structure=False):
        """ Make prediction given `src_seq` as input.

        Args:
            topics (list): list of topics (max 2) for the title. Use for contextual generation
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """

        torch.set_grad_enabled(False)
        text = []
        for tok in src_seq:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        if topics:
            topics = torch.LongTensor(topics).view(1,-1)
            if self.use_cuda:
                topics = topics.cuda()
        else:
            topics = None

        # If provided by the user, use that, else let it be the default max_length from training data.
        if max_length:
            self.model.decoder.max_length = max_length

        input_variable = torch.LongTensor(text).view(1, -1)
        if self.use_cuda:
            input_variable = input_variable.cuda()

        input_lengths = torch.LongTensor([len(src_seq)])

        prev_generated_seq = None
        structure_abstracts = None
        outputs = []
        for i in range(num_exams):
            _, _, other = \
                self.model(input_variable, prev_generated_seq, input_lengths, topics=topics, structure_abstracts=structure_abstracts)
            length = other['length'][0]

            tgt_id_seq = [other['sequence'][di][0].item() for di in range(length)]
            tgt_seq = [self.vectorizer.idx2word[tok] for tok in tgt_id_seq]
            output = ' '.join([i for i in tgt_seq if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])
            outputs.append(output)
            prev_generated_seq = torch.LongTensor(tgt_id_seq).view(1, -1).cuda() if self.use_cuda else torch.LongTensor(tgt_id_seq).view(1, -1)
            if use_structure:
                structure_abstracts = [other['gen_labels'][di] for di in range(length)]
                structure_abstracts = torch.LongTensor(structure_abstracts).view(1, -1).cuda() if self.use_cuda else torch.LongTensor(structure_abstracts).view(1, -1)
        return outputs

    def predict_batch(self, source, input_lengths, num_exams):
        torch.set_grad_enabled(False)
        output_seq = []
        input_variables = source
        for i in range(source.size(0)):
            title_id_seq = [input_variables[i][di].item() for di in range(input_lengths[i])]
            title_seq = [self.vectorizer.idx2word[tok] for tok in title_id_seq]
            title = ' '.join([k for k in title_seq if k != '<PAD>' and k != '<EOS>' and k != '<SOS>'])
            output_seq.append([title])
        prev_generated_seq = None
        for k in range(num_exams):
            _, _, other = \
                self.model(input_variables, prev_generated_seq, input_lengths)
            length = other['length']
            sequence = torch.stack(other['sequence'], 1).squeeze(2)
            prev_generated_seq = self._mask(sequence)
            for i in range(len(length)):
                opt_id_seq = [other['sequence'][di][i].item() for di in range(length[i])]
                opt_seq = [self.vectorizer.idx2word[tok] for tok in opt_id_seq]
                output = ' '.join([k for k in opt_seq if k != '<PAD>' and k != '<EOS>' and k != '<SOS>'])
                output_seq[i].append(output)
        return output_seq

    # Mask variable
    def _mask(self, prev_generated_seq):
        prev_mask = torch.eq(prev_generated_seq, 1).cpu().data.numpy()
        lengths = np.argmax(prev_mask,axis=1)
        max_len = prev_generated_seq.size(1)
        mask = []
        for i in range(prev_generated_seq.size(0)):
            if lengths[i] == 0:
                mask_line = [0] * max_len
            else:
                mask_line = [0] * lengths[i]
                mask_line.extend([1] * (max_len - lengths[i]))
            mask.append(mask_line)
        mask = torch.ByteTensor(mask)
        if self.use_cuda:
            mask = mask.cuda()
        return prev_generated_seq.data.masked_fill_(mask, 0)

    def preeval_batch(self, test_loader, abs_len, num_exams, use_topics=False, use_labels=False):
        torch.set_grad_enabled(False)
        org = {}
        refs = {}
        cands = []
        for i in range(num_exams):
            cands.append({})
        i = 0
        for batch_idx, data in enumerate(test_loader):
            topics = data[3] if use_topics else None
            # Since this is test time, we won't feed any structure labels from the test set.
            # The model has to figure them out for all the abstracts.
            structure_abstracts = None

            input_variables = data[0]
            target_variables = data[1]
            input_lengths = data[2]

            for j in range(input_variables.size(0)):
                i += 1
                ref = self.prepare_for_bleu(target_variables[j])
                refs[i] = [ref]
                org[i] = data[-1][j]
            prev_generated_seq = None
            for k in range(num_exams):
                _, _, other = \
                    self.model(input_variables, prev_generated_seq, input_lengths, topics=topics, structure_abstracts=structure_abstracts)
                length = other['length']
                sequence = torch.stack(other['sequence'], 1).squeeze(2)
                prev_generated_seq = self._mask(sequence)
                for j in range(len(length)):
                    out_seq = [other['sequence'][di][j] for di in range(length[j])]
                    out = self.prepare_for_bleu(out_seq)
                    cands[k][len(cands[k]) + 1] = out
                if use_labels:
                    structure_abstracts = torch.stack(other['gen_labels'], 1).squeeze(2)
            if i % 100 == 0:
                print("Percentages:  %.4f" % (i/float(abs_len)))
        return cands, refs, org


    def prepare_for_bleu(self, sentence):
        sent=[x.item() for x in sentence if x.item() != 0 and x.item() != 1 and x.item() != 2]
        sent = ' '.join([str(x) for x in sent])
        return sent

    def evaluate_abstracts(self, filename, vectorizer):
        refs = {}
        cand = {}
        with open(filename) as f:
            for i, line in enumerate(f):
                j = json.loads(line.strip())
                original_abstract = vectorizer.source_to_tokens(j["original"])
                generated_abstract = vectorizer.source_to_tokens(j["generated"])
                refs[i] = [self.prepare_for_bleu(original_abstract)]
                cand[i] = self.prepare_for_bleu(generated_abstract)
        return cand, refs

    def predict_seq_title(self, title, sec_seq, num_exams):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        torch.set_grad_enabled(False)
        text = []
        for tok in title:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        input_variable = torch.LongTensor(text).view(1, -1)
        if self.use_cuda:
            input_variable = input_variable.cuda()

        input_lengths = [len(title)]

        text = []
        for tok in sec_seq:
            if tok in self.vectorizer.word2idx:
                text.append(self.vectorizer.word2idx[tok])
            else:
                text.append(3)

        prev_generated_seq = torch.LongTensor(text).view(1, -1)
        if self.use_cuda:
            prev_generated_seq = prev_generated_seq.cuda()

        outputs = []
        for i in range(num_exams):
            _, _, other = \
                self.model(input_variable, prev_generated_seq, input_lengths)
            length = other['length'][0]

            tgt_id_seq = [other['sequence'][di][0].item() for di in range(length)]
            tgt_seq = [self.vectorizer.idx2word[tok] for tok in tgt_id_seq]
            output = ' '.join([i for i in tgt_seq if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])
            outputs.append(output)
            prev_generated_seq = torch.LongTensor(tgt_id_seq).view(1, -1)
            if self.use_cuda:
                prev_generated_seq = prev_generated_seq.cuda()
        return outputs
