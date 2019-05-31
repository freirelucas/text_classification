from __future__ import print_function
##############################################
# Section 0 - Load packages
# Sectio 1  - Data acquisition and cleaning
# Section 2 - Feature extraction - Find the best transformation in SVM

### Begin Section 0 ###

############################## utils
import pandas as pd
import numpy as np
import seaborn as sns
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
from sklearn.pipeline import Pipeline

######################### NLTK and stop words
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


pd.DataFrame(not_evidence_source_s).to_csv("dados")

for i in not_evidence_source_s:
    with open(folder_not_evidence+"/"+i) as f:
        not_evidence.append(f.read())

len(not_evidence)

# convert to data frame and make labels
data = pd.DataFrame(evidence)
data["Evidência"] = 1
data = pd.concat([pd.DataFrame(evidence_source),data], axis = 1)


data.columns = [ "Arquivo de origem", "Texto","Evidência"]
print(data.shape)
print(data.head(3))

print(data.Texto[0])
# now over not evidence cases
Z = pd.DataFrame(not_evidence)
Z["Evidência"] = int(0)
Z = pd.concat([pd.DataFrame(not_evidence_source_s),Z], axis = 1)
Z.columns = data.columns
print(Z.shape)
print(Z.head(5))

Z[:10]




# stack and shuffle
data = data.append(Z)
data.sample(frac=1).head(5)

from re import sub


# Define cleaning function
def cleaner(text):

    text = str(text)
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

# sanity check
print(X[0])

# define target in a Sklearn style
y = data.Evidência

# separa dados para cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


###  End Section 1 and begin Section 2 ###

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

### Lucas comment: just changed the input data



print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)),  # unigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (10, 50, 150), # from 80 to 150 due convergence problems
}


grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()

grid_search.fit(X, y)

print("done in %0.3fs" % (time() - t0))
print()
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# get  best_parameters without Stop words

vectorizer = TfidfVectorizer(grid_search.best_params_)
V = vectorizer.fit_transform(X)


def plot_embedding(V, y):
    """ Visualizes a vocabulary embedding via TSNE """
    V = TruncatedSVD(50).fit_transform(V)
    d = TSNE(metric='cosine').fit_transform(V)
    d = pd.DataFrame(d).assign(label = y.reset_index(drop=True))
    return sns.scatterplot(x = 0, y = 1, hue = 'label', data = d), d

ax, d = plot_embedding(V, y)

plt.savefig("Best_without_Stop_Words", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

print(grid_search.best_params_)

###### Comparing with Portuguese stop_words

vectorizer_stop = TfidfVectorizer(grid_search.best_params_, stop_words= nltk.corpus.stopwords.words('portuguese'))
V_stop = vectorizer.fit_transform(X)


ax, d = plot_embedding(V_stop, y)


plt.savefig("Best_with_Stop_Words", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


best_alpha = grid_search.best_params_[ 'clf__alpha']
best_penalty = grid_search.best_params_[ 'clf__penalty']
best_max_iter = grid_search.best_params_[ 'clf__max_iter']

clf = SGDClassifier(alpha = best_alpha, penalty = best_penalty, max_iter = best_max_iter)
target_names = [ "Irrelevante", "Evidência"]

print ("Performance sem stop-words")
clf.fit(V, y)
y_pred = clf.predict(V)
print(metrics.classification_report(y, y_pred, target_names=target_names))

print ("Performance com Stop-words")
clf.fit(V_stop, y)
y_pred_stop = clf.predict(V_stop)
print(metrics.classification_report(y, y_pred, target_names=target_names))
