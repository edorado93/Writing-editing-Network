import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from gensim.models import KeyedVectors
import re
import json

#provide pretrained embeddings for text
def load_embeddings(pytorch_embedding, word2idx, filename, embedding_size):
    print("Copying pretrained word embeddings from ", filename, flush=True)
    en_model = KeyedVectors.load_word2vec_format(filename)
    """ Fetching all of the words in the vocabulary. """
    pretrained_words = set()
    for word in en_model.vocab:
        pretrained_words.add(word)

    arr = [0] * len(word2idx)
    for word in word2idx:
        index = word2idx[word]
        if word in pretrained_words:
            arr[index] = en_model[word]
        else:
            arr[index] = np.random.uniform(-1.0, 1.0, embedding_size)

    """ Creating a numpy dictionary for the index -> embedding mapping """
    arr = np.array(arr)
    """ Add the word embeddings to the empty PyTorch Embedding object """
    pytorch_embedding.weight.data.copy_(torch.from_numpy(arr))
    return pytorch_embedding

def plot_topical_encoding(topic_dictionary, topical_embeddings, tensorboardX_writer, step):
    topics = []
    embeddings = []
    for t, index in topic_dictionary.items():
        topics.append(t)
        embeddings.append(topical_embeddings(torch.tensor([index], dtype=torch.long).cuda()))
    tensorboardX_writer.add_embedding(torch.squeeze(torch.stack(embeddings)), topics, global_step=step)

#Transforms a Corpus into lists of word indices.
class Vectorizer:
    def __init__(self, max_words=None, min_frequency=None, start_end_tokens=True, maxlen=None):
        self.vocabulary = None
        self.vocabulary_size = 0
        self.word2idx = dict()
        self.idx2word = dict()
        #most common words
        self.max_words = max_words
        #least common words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.maxlen = maxlen
        self.context_vectorizer = {}

    def topics_to_index_tensor(self, topics):
        vec = [self.context_vectorizer[t] if t in self.context_vectorizer else self.context_vectorizer["algorithm"] for t in topics]
        return vec

    def _find_max_sentence_length(self, corpus, template):
        if not template:
            self.maxlen = max(len(sent) for document in corpus for sent in document)
        else:
            self.maxlen = max(len(sent) for sent in corpus)
        if self.start_end_tokens:
            self.maxlen += 2

    def _build_vocabulary(self, corpus, template):
        if not template:
            vocabulary = Counter(word for document in corpus for sent in document for word in sent)
        else:
            vocabulary = Counter(word for sent in corpus for word in sent)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary) + 2  # padding and unk tokens
        if self.start_end_tokens:
            self.vocabulary_size += 2

    def _build_word_index(self):
        self.word2idx['<UNK>'] = 3
        self.word2idx['<PAD>'] = 0

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 1
            self.word2idx['<SOS>'] = 2

        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def fit(self, corpus, template = False):
        if not self.maxlen:
            self._find_max_sentence_length(corpus, template)
        self._build_vocabulary(corpus, template)
        self._build_word_index()

    def add_start_end(self, vector):
        vector.append(self.word2idx['<EOS>'])
        return [self.word2idx['<SOS>']] + vector

    def transform_sentence(self, sentence):
        """
        Vectorize a single sentence
        """
        vector = [self.word2idx.get(word, 3) for word in sentence]
        if self.start_end_tokens:
            vector = self.add_start_end(vector)
        return vector

    def transform(self, corpus, template = False):
        """
        Vectorizes a corpus in the form of a list of lists.
        A corpus is a list of documents and a document is a list of sentence.
        """
        vcorpus = []
        if not template:
            for document in corpus:
                vcorpus.append([self.transform_sentence(sentence) for sentence in document])
        else:
            vcorpus.extend([self.transform_sentence(sentence) for sentence in corpus])
        return vcorpus

class headline2abstractdataset(Dataset):

    structure_dict = {"introduction": 1, "body": 2, "conclusion": 3}

    def __init__(self, path, vectorizer, USE_CUDA=torch.cuda.is_available(), max_len=200):
        self.head_len = 0
        self.abs_len = 0
        self.max_len = max_len
        self.max_context_length = 1
        self.vectorizer = vectorizer
        self.corpus, self.topics_corpus, self.abstract_structures = self._read_corpus(path)
        self.data = self._vectorize_corpus()
        self._initalcorpus()
        self.USE_CUDA = USE_CUDA

    def pad_sentence_vector(self, vector, maxlen, pad_value=0):
        org_length = len(vector)
        padding = maxlen - org_length
        vector.extend([pad_value] * padding)
        return vector

    def _initalcorpus(self):
        old = []
        for i, j, st in zip(self.data, self.topics_corpus, self.abstract_structures):
            source = i[0]
            target = i[1]
            vectorized_topics = j
            if len(source) > self.head_len:
                self.head_len = len(source)
            if len(target) <= self.max_len:
                if len(target) > self.abs_len:
                    self.abs_len = len(target)
            else:
                target = target[:self.max_len-1]
                target.append(1)#word2idx['<EOS>'] = 1
                self.abs_len = len(target)
            old.append((source[1:-1], target, vectorized_topics, st))
        old.sort(key = lambda x: len(x[0]), reverse = True)
        corpus = []
        for source, target, vectorized_topics, structure_abstract in old:
            vectorized_topics = self.pad_sentence_vector(vectorized_topics, self.max_context_length, pad_value=self.vectorizer.context_vectorizer['algorithm'])
            team = [len(source), len(target), self.pad_sentence_vector(source, self.head_len), self.pad_sentence_vector(target, self.abs_len), vectorized_topics, structure_abstract]
            corpus.append(team)
        self.data = corpus

    def _read_corpus(self, path):
        abstracts = []
        headlines = []
        topics = []
        i = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                headlines.append(j["title"])
                abstracts.append(j["abstract"])
                if "topics" in j:
                    topics.append(j["topics"])
                i += 1
        corpus = []
        topics_v = []
        abstract_structures = []
        self.vectorizer.context_vectorizer['algorithm'] = 0
        for i in range(len(abstracts)):
            if len(headlines[i]) > 0 and len(abstracts[i]) > 0:
                h_a_pair = []
                h_a_pair.append(self._tokenize_word(headlines[i]))

                # Assumption is that every word will have structural information associated with it.
                abstract_and_structure_information = self._tokenize_word(abstracts[i])
                structure = [headline2abstractdataset.structure_dict[s] for s, _ in abstract_and_structure_information]
                abstract = [a for _, a in abstract_and_structure_information]
                h_a_pair.append(abstract)
                # This list `structure` has vectorised structure labels. The labels can be one of
                # introduction, body or conclusion. Correspondingly, we have indexes of 1, 2 or 3 in
                # the vectorizer.context_vectorizer.
                abstract_structures.append(structure)
                if len(h_a_pair) > 1:
                    corpus.append(h_a_pair)
                    vectorized_topics = []
                    if topics:
                        for t in topics[i]:
                            t = t.lower()
                            if t not in self.vectorizer.context_vectorizer:
                                self.vectorizer.context_vectorizer[t] = len(self.vectorizer.context_vectorizer)
                            vectorized_topics.append(self.vectorizer.context_vectorizer[t])
                        self.max_context_length = max(self.max_context_length, len(vectorized_topics))
                    topics_v.append(vectorized_topics)
        return corpus, topics_v, abstract_structures

    def _tokenize_word(self, sentence):
        result = []
        for word in sentence.split():
            if word:
                result.append(word)
        return result

    #sentence to word id
    def _vectorize_corpus(self):
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(self.corpus)
        return self.vectorizer.transform(self.corpus)

    def __getitem__(self, index):
        len_s, len_t, source, target, topics, structure_abstracts = self.data[index]
        source = torch.LongTensor(source)
        topics = torch.LongTensor(topics)
        target = torch.LongTensor(target)
        structure_abstracts = torch.LongTensor(structure_abstracts)
        if self.USE_CUDA:
            source = source.cuda()
            target = target.cuda()
            topics = topics.cuda()
            structure_abstracts = structure_abstracts.cuda()
        return source, target, len_s, topics, structure_abstracts

    def __len__(self):
        return len(self.data)
