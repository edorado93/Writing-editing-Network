import sys
import torch.nn as nn
import torch
from torch.backends import cudnn
from flask import Flask, render_template, request, jsonify
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
sys.path.insert(0, "network/")
from time import  sleep
from network.predictor import Predictor
from network import configurations

testing_mode = True
test_topics = ["Topic"+str(i+1) for i in range(76)]

def unknown_word_embeddings():
    filename = "embeddings/wiki-news-300d-1M.vec"
    gen_model = KeyedVectors.load_word2vec_format(filename)
    pretrained_words = set()
    for word in gen_model.vocab:
        pretrained_words.add(word)
    return pretrained_words, gen_model

def replace_unknown_words(title_words):
    new_title = []
    print(title_words)
    for word in title_words:
        if word not in custom_words:
            if word not in general_words:
                new_title.append(word)
            else:
                gen_emb = gen_model[word].reshape(1, -1)
                cosine = -1
                replacement = None
                for science_word in custom_words:
                    val = cosine_similarity(custom_model[science_word].reshape(1, -1), gen_emb)
                    if val > cosine:
                        cosine = val
                        replacement = science_word
                print("Replaced {} with {}".format(word, replacement))
                new_title.append(replacement)
        else:
            new_title.append(word)
    return new_title

def load_pretrained_ARXIV_embeddings():
    # Loading PreTrained Word Embeddings. Will be used to find out 10 relevant topics
    filename = "embeddings/complete.vec"
    custom_model = KeyedVectors.load_word2vec_format(filename)
    custom_words = set()
    for word in custom_model.vocab:
        custom_words.add(word)
    return custom_words, custom_model

def topical_word_embeddings():
    topics_emb = []
    # Read topics line by line and obtain average word embeddings for each of the topical phrase.
    with open("top-76-relevant-topics.txt") as f:
        for line in f:
            t = line.strip().split()
            t_emb = []
            for atom in t:
                if atom in custom_words:
                    t_emb.append(custom_model[atom])
            if t_emb:
                topics_emb.append((np.array(sum(t_emb) / len(t_emb)).reshape(1, -1), " ".join(t)))
    return topics_emb

def get_topics_according_to_relevance(title):
    title = title.split()
    title_embedding = []
    for t in title:
        if t in custom_words:
            title_embedding.append(custom_model[t])
    title_embedding = np.array(sum(title_embedding) / len(title_embedding)).reshape(1,-1)

    ans = []
    for topic_emb, topic in topical_embeddings:
        ans.append((topic, cosine_similarity(title_embedding, topic_emb)))

    ans = sorted(ans, key=lambda x: x[1], reverse=True)
    return [a[0] for a in ans]

if not testing_mode:
    from network import main

    use_cuda = torch.cuda.is_available()

    # The configuration to load for the model
    conf = "l4"

    # Initialize the bare bones model
    config, model, _, _, vectorizer = main.init(conf, 1111, use_cuda)[0:5]

    # Loading the saved model which will now be used for generation.
    save = "models/"+config.experiment_name + '.pkl'
    print("==> Loading checkpoint", save)
    checkpoint = torch.load(save)
    model.load_state_dict(checkpoint['state_dict'])
    print("==> Successfully Loaded checkpoint", save)

    # Load the predictor object that shall be used for generation.
    predictor = Predictor(model, vectorizer, use_cuda=use_cuda)

    # Load fastText pretrained word embeedings for unknown word replacements.
    general_words, gen_model = unknown_word_embeddings()

    # Load set of words and their embeddings in our pretrained WE.
    custom_words, custom_model = load_pretrained_ARXIV_embeddings()

    # Obtain average word embeddings for each of the 76 titles that we have.
    topical_embeddings = topical_word_embeddings()

    print("Loaded pretrained word embeddings")

# Defining the flask app
app = Flask(__name__)

"""   Flask APIs Begin """

@app.route('/getTopics', methods=['POST'])
def get_topics():
    data = request.get_json()
    title_entered_by_user = data["title"]
    topics = get_topics_according_to_relevance(title_entered_by_user) if not testing_mode else test_topics
    return jsonify({"topics": topics})

@app.route('/getAbstracts', methods=['POST'])
def get_abstract():
    data = request.get_json()
    title_entered_by_user = data["title"]
    topics_entered = data["topics"]
    seq = title_entered_by_user.strip().split(' ')
    seq = replace_unknown_words(seq)
    print("New title", seq)
    topics = vectorizer.topics_to_index_tensor(topics_entered)
    num_exams = 2
    max_length = 200
    print("Generating now")
    outputs = predictor.predict(seq, num_exams, max_length=max_length, topics=topics, use_structure=config.use_labels)
    return jsonify({"abstract": outputs[1]})

@app.route('/')
def root():
    return render_template('index.html')
