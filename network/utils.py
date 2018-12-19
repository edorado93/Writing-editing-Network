import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from gensim.models import KeyedVectors
import subprocess
import json

def get_gpu_memory_map():
    """Get the current gpu usage.
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')][:torch.cuda.device_count()]
    gpu_memory_map = dict(zip(range(torch.cuda.device_count()), gpu_memory))
    return gpu_memory_map

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

#Transforms a Corpus into lists of word indices.
class Vectorizer:
    def __init__(self, max_words=None, min_frequency=None, start_end_tokens=True, maxlen=None):
        self.vocabulary = None
        self.word2idx = dict()
        self.idx2word = dict()
        #most common words
        self.max_words = max_words
        #least common words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.context_vectorizer = {}

    def topics_to_index_tensor(self, topics):
        vec = [self.context_vectorizer[t] if t in self.context_vectorizer else self.context_vectorizer["algorithm"] for t in topics]
        return vec

    def source_to_tokens(self, src_seq):
        text = []
        src_seq = src_seq.strip().split(' ')
        for tok in src_seq:
            if tok in self.word2idx:
                tok = self.word2idx[tok]
            else:
                # UNK token
                tok = 3
            text.append(torch.LongTensor([tok]).view(1,1))
        return text

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

    def _build_word_index(self):
        self.word2idx['<UNK>'] = 3
        self.word2idx['<PAD>'] = 0

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 1
            self.word2idx['<SOS>'] = 2

        for idx, word in enumerate(self.vocabulary):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def fit(self, corpus, template = False):
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
    def __init__(self, path, vectorizer, USE_CUDA=False, max_len=200, use_topics=False, use_structure_info=False):
        self.use_topics = use_topics
        self.use_structure_info = use_structure_info
        self.head_len = 0
        self.abs_len = 0
        self.max_len = max_len
        self.max_context_length = 1
        self.untokenized_original = {}
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
        for i, content in enumerate(self.data):
            source = content[0]
            target = content[1]
            contextual_dictionary = {}

            if self.use_topics:
                contextual_dictionary["topics"] = self.topics_corpus[i]
            if self.use_structure_info:
                contextual_dictionary["structure"] = self.abstract_structures[i]

            if len(source) > self.head_len:
                self.head_len = len(source)
            if len(target) <= self.max_len:
                if len(target) > self.abs_len:
                    self.abs_len = len(target)
            else:
                target = target[:self.max_len-1]
                target.append(1)#word2idx['<EOS>'] = 1
                self.abs_len = len(target)

            old.append((source[1:-1], target, contextual_dictionary, i))
        old.sort(key=lambda x: len(x[0]), reverse = True)
        corpus = []
        for source, target, contextual_dictionary, original_index in old:

            if self.use_topics:
                contextual_dictionary["topics"] = self.pad_sentence_vector(contextual_dictionary["topics"], self.max_context_length, pad_value=self.vectorizer.context_vectorizer['algorithm'])
            if self.use_structure_info:
                contextual_dictionary["structure"] = self.pad_sentence_vector(contextual_dictionary["structure"], self.abs_len, pad_value=3)
            team = [len(source), len(target), self.pad_sentence_vector(source, self.head_len), self.pad_sentence_vector(target, self.abs_len), contextual_dictionary, original_index]
            corpus.append(team)
        self.data = corpus

    def _read_corpus(self, path):
        abstracts = []
        headlines = []
        topics = []
        labels = []
        i = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                headlines.append(j["title"])
                abstracts.append(j["abstract"])
                self.untokenized_original[i] = (j["title"], j["abstract"])
                if "topics" in j:
                    topics.append(j["topics"])
                if "labels" in j:
                    labels.append(j["labels"])
                i += 1
        self.vectorizer.context_vectorizer['algorithm'] = 0
        self.vectorizer.context_vectorizer['introduction'] = 1
        self.vectorizer.context_vectorizer['body'] = 2
        self.vectorizer.context_vectorizer['conclusion'] = 3
        corpus = self._read_data(headlines, abstracts)
        topics_v = self._read_topics(topics)
        abstract_structures = self._read_structure(labels)
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

    def _read_data(self, headlines, abstracts):
        corpus = []
        for i in range(len(abstracts)):
            if len(headlines[i]) > 0 and len(abstracts[i]) > 0:
                h_a_pair = []
                h_a_pair.append(self._tokenize_word(headlines[i]))
                h_a_pair.append(self._tokenize_word(abstracts[i]))
                if len(h_a_pair) > 1:
                    corpus.append(h_a_pair)
        return corpus

    def _read_topics(self, topics):
        topics_v = []
        for i in range(len(topics)):
            vectorized_topics = []
            if topics:
                for t in topics[i]:
                    t = t.lower()
                    if t not in self.vectorizer.context_vectorizer:
                        self.vectorizer.context_vectorizer[t] = len(self.vectorizer.context_vectorizer)
                    vectorized_topics.append(self.vectorizer.context_vectorizer[t])
                self.max_context_length = max(self.max_context_length, len(vectorized_topics))
            topics_v.append(vectorized_topics)
        return topics_v

    def _read_structure(self, structure_info):
        structure = []
        for i in range(len(structure_info)):
            tokenised = [self.vectorizer.context_vectorizer[s] for s in structure_info[i]]
            if self.vectorizer.start_end_tokens:
                tokenised.append(self.vectorizer.context_vectorizer['conclusion'])
            tokenised = [self.vectorizer.context_vectorizer['introduction']] + tokenised
            structure.append(tokenised)

        return structure

    def __getitem__(self, index):
        len_s, len_t, source, target, context_dictionary, original_index = self.data[index]
        source = torch.LongTensor(source).cuda() if self.USE_CUDA else torch.LongTensor(source)
        target = torch.LongTensor(target).cuda() if self.USE_CUDA else torch.LongTensor(target)
        org_index = torch.LongTensor([original_index]).cuda() if self.USE_CUDA else torch.LongTensor([original_index])
        ret = [source, target, len_s]
        if self.use_topics:
            topics = (torch.LongTensor(context_dictionary["topics"]).cuda() if self.USE_CUDA else torch.LongTensor(context_dictionary["topics"])) if self.use_topics else None
            ret.append(topics)
        if self.use_structure_info:
            structure_abstracts = (torch.LongTensor(context_dictionary["structure"]).cuda() if self.USE_CUDA else torch.LongTensor(context_dictionary["structure"])) if self.use_structure_info else None
            ret.append(structure_abstracts)
        ret.append(org_index)
        return ret

    def __len__(self):
        return len(self.data)
