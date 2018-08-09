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
# from network import main

def load_pretrained_ARXIV_embeddings():
    # Loading PreTrained Word Embeddings. Will be used to find out 10 relevant topics
    filename = "embeddings/complete-512.vec"
    en_model = KeyedVectors.load_word2vec_format(filename)
    pretrained_words = set()
    for word in en_model.vocab:
        pretrained_words.add(word)
    return pretrained_words, en_model

def topical_word_embeddings():
    topics_emb = []
    # Read topics line by line and obtain average word embeddings for each of the topical phrase.
    with open("top-76-relevant-topics.txt") as f:
        for line in f:
            t = line.strip().split()
            t_emb = []
            for atom in t:
                if atom in pretrained_words:
                    t_emb.append(en_model[atom])
            if t_emb:
                topics_emb.append((np.array(sum(t_emb) / len(t_emb)).reshape(1, -1), " ".join(t)))
    return topics_emb

def get_top_10_relevant_topics(title):
    title = title.split()
    title_embedding = []
    for t in title:
        if t in pretrained_words:
            title_embedding.append(en_model[t])
    title_embedding = np.array(sum(title_embedding) / len(title_embedding)).reshape(1,-1)

    ans = []
    for topic_emb, topic in topical_embeddings:
        ans.append((topic, cosine_similarity(title_embedding, topic_emb)))

    ans = sorted(ans, key=lambda x: x[1], reverse=True)
    return [a[0] for a in ans[:10]]

# # Load set of words and their embeddings in our pretrained WE.
# pretrained_words, en_model = load_pretrained_ARXIV_embeddings()
#
# # Obtain average word embeddings for each of the 76 titles that we have.
# topical_embeddings = topical_word_embeddings()
#
# # The configuration to load for the model
# config = main.config
#
# # Loading the saved model which will now be used for generation.
# model = main.load_checkpoint("models/"+config.experiment_name + '.pkl', model)

# Defining the flask app
app = Flask(__name__)

"""   Flask APIs Begin """

@app.route('/getTopics', methods=['POST'])
def get_topics():
    data = request.get_json()
    title_entered_by_user = data["title"]
    # topics = get_top_10_relevant_topics(title_entered_by_user)
    topics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    return jsonify({"topics": topics})

@app.route('/getAbstracts', methods=['POST'])
def get_abstract():
    data = request.get_json()
    title_entered_by_user = data["title"]
    topics_entered = data["topics"]
    sleep(4);
    return jsonify({"abstract": "Hello My name is Sachin Malhotra and this is an abstract"*5})


@app.route('/')
def root():
    return render_template('index.html')
