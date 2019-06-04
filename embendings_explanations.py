from __future__ import print_function
##############################################
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
import time


############################## sk learn
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

######################### NLP
import nltk
from nltk.corpus import stopwords

start_time = time.time()
folder_evidence = "../evidence"
folder_not_evidence = "../not_evidence"

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


not_evidence =[]
for i in not_evidence_source:
    with open(folder_not_evidence+"/"+i) as f:
        not_evidence.append(f.read())


# convert to data frame and make labels
data = pd.DataFrame(evidence)
data[1] = 1
data = pd.concat([pd.DataFrame(evidence_source),data], axis = 1)

data.columns = [ "Arquivo de origem", "Texto","Evidência"]


# now over not evidence cases
Z = pd.DataFrame(not_evidence)
Z[1] = 0
Z = pd.concat([pd.DataFrame(not_evidence_source),Z], axis = 1)
Z.columns = data.columns

# stack and shuffle
data = data.append(Z)
data.sample(frac=1).head(5) #sanity check

from re import sub
# Define cleaning function
def cleaner(text):
    text = str(text)
    text = text.lower()
    text = sub("\n","",text) # remove line break
    text = sub(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}',"",text)
    text = sub(".com","",text) # remove part of sites
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


# compute class imbalance

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y), y)
print("Class imbalance: {}".format(class_weights))
#prepare stop words
pt_stop_words = nltk.corpus.stopwords.words('portuguese')

########### word embeddings

#source = "/media/lucas/LINUX_DATA/"

# setting up a transfer representation prepared by University of São Paulo http://nilc.icmc.usp.br/embeddings

#from gensim.models import KeyedVectors
#from gensim.models.doc2vec import Doc2VecKeyedVectors


# Cbow and skip-gran 100 dimension vectors pretrained
#model_cbow = KeyedVectors.load_word2vec_format(source+"cbow_s100.txt")
#model_skip =KeyedVectors.load_word2vec_format(source+"skip_s100.txt")

################## classifier
#Simple embeddings
vec_tfidf = TfidfVectorizer(stop_words = pt_stop_words)
vec_count = CountVectorizer(stop_words = pt_stop_words)

# dimensionality reduction for simple latent semantic analysis
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
lsa_tfidf = make_pipeline(vec_tfidf, svd)
lsa_count = make_pipeline(vec_count, svd)

# instanciate Ramdom forest

clf=RandomForestClassifier(n_estimators=100, class_weight="balanced")


# prepare pipelines

pipe_tfidf = make_pipeline(lsa_tfidf, clf)
pipe_count = make_pipeline(lsa_count, clf)


################# Confusion metrics // recall analysis and vizualization



def print_results (model, labels = ["Irrelevante","Evidência"]):
    print("##### Results from model: {}".format(model))
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ax= sns.heatmap(cm, annot=True, cmap="Blues")#annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig("confusion_matrix.png")
    plt.show()




############### Explanations

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick

def export_explanations (model, X_test = X_test, class_names = ["Irrelevante", "Evidência"], name="name", n_exp=20):
    explainer = LimeTextExplainer(class_names=class_names)
    idx = int(random.random()) # sanity check
    exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=20)
    print('Document id: %d' % idx)
    print('Probability(Evidência) =', model.predict_proba([X_test[idx]])[0,1])
    print('True class: %s' % class_names[y_test.iloc[idx]])
#########   SUBMODULAR PICK
    sp_obj = submodular_pick.SubmodularPick(explainer, X_test, model.predict_proba, sample_size=n_exp, num_features=15,num_exps_desired=n_exp)
    imagens = [exp.as_pyplot_figure(label=exp.available_labels()[1]) for exp in sp_obj.sp_explanations]
    i =0
    for exp in sp_obj.sp_explanations:
        exp.save_to_file(file_path="{}explanation{}.html".format(name,i))
        i+=1
    i=0
    for img in imagens:
        img.savefig("{}Imagem{}".format(name,i))
        i+=1
    pass



# fit, evalute and explain models
#pipe_tfidf.fit(X_train, y_train)
#print(pipe_tfidf.score(X_test, y_test))
#print_results(pipe_tfidf)
#export_explanations(pipe_tfidf, name="Tfidf")

pipe_count.fit(X_train, y_train)
print(pipe_count.score(X_test, y_test))
print_results(pipe_count)
export_explanations(pipe_count, name="CountVectorizer")


##########
end_time = (time.time() -start_time)/60
print("This code code runs in {} minutes".format(end_time))
