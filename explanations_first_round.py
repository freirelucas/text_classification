from __future__ import print_function
##############################################
# Section 0 - Load packages
# Sectio 1  - Data acquisition and cleaning
# Section 2 - Featur extraction - Find the best transformation in SVM
# Section 3 - Get the best transformation and tune a family of classifiers
# Section 4 - Analyse Explainability with LIME and Eli5
# Section 5 - Analyse performance improvement after human-treatment of Lime output

### Begin Section 0 ###

############################## utils
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
import matplotlib.pyplot as plt
import os
from itertools import combinations
from re import sub, split
from pprint import pprint
from time import time
import logging
import random

############################## sk learn
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

######################### NLP
import nltk
from nltk.corpus import stopwords


folder_evidence = "/home/lucas/Desktop/master_thesis/evidence"
folder_not_evidence = "/home/lucas/Desktop/master_thesis/not_evidence"

# collect evidence files names
evidence_source = []
for file in os.listdir(folder_evidence):
    if file.endswith(".txt"):
        #print(os.path.join(folder, file))
        evidence_source.append(os.path.join(file))


# collect NOT evidence file names
not_evidence_source = []
for file in os.listdir(folder_not_evidence):
    if file.endswith(".txt"):
        #print(os.path.join(folder, file))
        not_evidence_source.append(os.path.join(file))

# open and read files
evidence =[]
for i in evidence_source:
    with open(folder_evidence+"/"+i) as f:
        evidence.append(f.read())

n_samp = int(len(not_evidence_source)/20)

not_evidence =[]
not_evidence_source_s = random.sample(not_evidence_source, n_samp )
for i in not_evidence_source_s:
    with open(folder_not_evidence+"/"+i) as f:
        not_evidence.append(f.read())

len(not_evidence_source_s)

# convert to data frame and make labels
data = pd.DataFrame(evidence)
data[1] = 1
data = pd.concat([pd.DataFrame(evidence_source),data], axis = 1)



data.columns = [ "Arquivo de origem", "Texto","Evidência"]


# now over not evidence cases
Z = pd.DataFrame(not_evidence)
Z[1] = 0
Z = pd.concat([pd.DataFrame(not_evidence_source_s),Z], axis = 1)
Z.columns = data.columns






# stack and shuffle
data = data.append(Z)
data.sample(frac=1).head(5)

from re import sub
# Define cleaning function
def cleaner(text):

    text = text.lower()
    text = sub("\n","",text) # remove line break
    text = sub(".com","",text) # remove part of email/sites
    text = sub("@\w+","",text) # remove part of email
    text = sub("www","",text) # remove part of sites
    text = sub("mailto","",text) # remove non-words identified eyebailing the features set
    text = sub("gmail","",text) # same
    text = sub("para","",text) # same
    text = ' '.join(text.split()) # Extra Whitespace

    return text

# call the cleaner
X = []
X = [cleaner(str(text)) for text in data.Texto]


# define target in a Sklearn style
y = data.Evidência

# separa dados para cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#prepare stop words
pt_stop_words = nltk.corpus.stopwords.words('portuguese')


########### word embeddings

#source = "/media/lucas/LINUX_DATA/"

# setting up a transfer representation prepared by University of São Paulo http://nilc.icmc.usp.br/embeddings
#from gensim.models import KeyedVectors
#model_cbow = KeyedVectors.load_word2vec_format(source+"cbow_s100.txt")
#model_skip =KeyedVectors.load_word2vec_format(source+"skip_s100.txt")


#class TfidfEmbeddingVectorizer(object):
#    def __init__(self, word2vec):
#        self.word2vec = word2vec
#        self.word2weight = None
#        self.dim = len(word2vec.itervalues().next())
#
#    def fit(self, X, y):
#        tfidf = TfidfVectorizer(analyzer=lambda x: x)
#        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
#        max_idf = max(tfidf.idf_)
#        self.word2weight = defaultdict(
#            lambda: max_idf,
#            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

#        return self

#    def transform(self, X):
#        return np.array([
#                np.mean([self.word2vec[w] * self.word2weight[w]
#                         for w in words if w in self.word2vec] or
#                        [np.zeros(self.dim)], axis=0)
#                for words in X
#            ])



################## classifier
vec = TfidfVectorizer(stop_words = pt_stop_words)
# dimensionality reduction before

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)

lsa = make_pipeline(vec, svd)
clf = SVC(kernel='rbf', probability=True)
pipe = make_pipeline(lsa, clf)
pipe.fit(X_train, y_train)

print(pipe.score(X_test, y_test))

y_pred = pipe.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))

################# Confusion metrics






############### Explanations

import eli5
from eli5.lime.lime import TextExplainer
# start the explainer function
te = TextExplainer(random_state=42)

# set up randon instances to analyse
import random
indexes_evidence = np.random.choice(len(evidence), size=(10))
indexes_not_evidence = np.random.choice(len(not_evidence_source_s), size=(10)) + len(evidence)

# use Eli5 to export Html explanatons objects
from eli5.formatters import format_as_html
def save_explanations(i):
    # first train the explainer
    te.fit(X[i], pipe.predict_proba)
    # save the table of weights
    z = format_as_html(te.explain_weights(top=20))
    Html_file= open("Evidence_idx{}_explain_weights.html".format(i),"w")
    Html_file.write(z)
    Html_file.close
    # save the document explained
    z = format_as_html(te.explain_prediction())
    Html_file= open("Evidence_idx{}_explain_prediction.html".format(i),"w")
    Html_file.write(z)
    Html_file.close

import multiprocessing as mp
pool = mp.Pool(processes=8)
# Submit jobs in parallel
pool.map_async(save_explanations, indexes_evidence)
pool.close()
