import gensim
from nltk.tokenize import word_tokenize
import json
import operator
import random

"""
    Code from 
    https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
"""

# Let's create some documents. *********************
raw_documents = []
original = []
with open("data.txt") as f:
    for line in f:
        j = json.loads(line)
        original.append(j)
        raw_documents.append(j["abstract"])
print("Number of documents:",len(raw_documents))


# Obtain NLTK tokenized words for each abstract *********************
gen_docs = [t.split() for t in raw_documents]

# We will create a dictionary from a list of documents *********************
dictionary = gensim.corpora.Dictionary(gen_docs)

# Now we will create a corpus. A corpus is a list of bags of words *********************
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

# Now we create a tf-idf model from the corpus *********************
tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

# Now we will create a similarity measure object in tf-idf space *********************
sims = gensim.similarities.Similarity('/Users/sachinmalhotra/find-similar',tf_idf[corpus], num_features=len(dictionary))
print(sims)


additional_found = 0
with open("additional_topics_added.txt", "w") as f:
    for i, j in enumerate(original):
        query_abstract = j["abstract"].split()
        query_abstract_bow = dictionary.doc2bow(query_abstract)
        query_abstract_tf_idf = tf_idf[query_abstract_bow]
        similarities = sims[query_abstract_tf_idf]
        sim_and_index = [(s, i) for i, s in enumerate(similarities)]
        sim_and_index.sort(reverse=True, key=lambda x: x[0])
        top_5 = [i for s,i in sim_and_index[:10]]

        own_topics = j["topics"]
        topics = {}
        # 0th index document is self.
        for t in top_5[1:]:
            t = original[t]
            for context in t["topics"]:
                if context not in own_topics:
                    topics[context] = topics.get(context, 0) + 1

        if topics:
            additional_found += 1
            j["topics"].append(max(topics.items(), key=operator.itemgetter(1))[0])
        f.write(json.dumps(j)+"\n")

        if (i+1) % 1000 == 0:
            print("Processed {} abstracts".format(i+1))

print("Out of {} abstracts, we found an additional topic for {}".format(len(original), additional_found))