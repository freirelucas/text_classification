"""
Done:
    Upweight the most important words (partial -> to load data, vectorizer -> raw docs)
    DBSCAN A) unlebeled, B)labeled, C) all
    Save/Export -> logs  1) F1 and recall, 2) DBSCAN
    SGD classifier fixed (logistic aproximation of probability - > just used the log loss with kernel aproximation)
    Export also responsive explanation (not only submodular_pick)
    Upweight the most important words (as pipe to LIME)
    Predict 20 most likely docs (done) and explain it

To do:


    LDA / top 20 words each topic - DBSCAN to discard labeled as irrelevant groups
    Canonical logging


"""


from __future__ import print_function
##############################################
############################## utils
import pandas as pd
import itertools
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
import sys
from  datetime import date, datetime
import matplotlib as mat_plt
mat_plt.rcParams.update({'figure.max_open_warning': 0})


############################## sk learn
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.multiclass import unique_labels

######################### NLP
import nltk
from nltk.corpus import stopwords
log = []
start_time = time.time()
folder_evidence = "../evidence"
folder_not_evidence = "../not_evidence"
folder_unknown = "../unknown"

# collect evidence files names
evidence_source = []
for file in os.listdir(folder_evidence):
    if file.endswith(".txt"):
        evidence_source.append(os.path.join(file))


# open and read files
evidence =[]
for i in evidence_source:
    with open(folder_evidence+"/"+i) as f:
        evidence.append(f.read())

# collect NOT evidence file names
not_evidence_source = []
for file in os.listdir(folder_not_evidence):
    if file.endswith(".txt"):
        not_evidence_source.append(os.path.join(file))

# open and read files
not_evidence =[]
for i in not_evidence_source:
    with open(folder_not_evidence+"/"+i) as f:
        not_evidence.append(f.read())

# collect unknown file names
unknown_source = []
for file in os.listdir(folder_unknown):
    if file.endswith(".txt"):
        unknown_source.append(os.path.join(file))

path_evidence = [folder_evidence+"/"+a for a in evidence_source]
path_not_evidence = [folder_not_evidence+"/"+a for a in not_evidence_source]
path_unknown = [folder_unknown+"/"+a for a in unknown_source]
path_labeled = path_evidence + path_not_evidence
# prepare the
complete_path = path_evidence + path_not_evidence + path_unknown

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

#######################################################
''' For labeled data we are loading in pandas DF
For files of unlabeld data we don't want to store the text in memory Ram

'''
#######################################################

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
    text = ' '.join(text.split(" ")) # Extra Whitespace

    return text

# call the cleaner
X = []
X = [cleaner(str(text)) for text in data.Texto]


# define target with labeled data
y = data.Evidência.reset_index(drop=True)

#prepare stop words in portuguese

pt_stop_words = nltk.corpus.stopwords.words('portuguese')


#######################################################
'''
Building the vectorizer

Steps 1) learn the vocabulary from labeled data  2) upweight important terms  3) tf-idf transform
After first iteration start to feed the black list and the white list

'''
#######################################################
blacklist = ['eita']                           # read from a csv after the first run white list
pt_stop_words = pt_stop_words + blacklist      # update stop words

vectorizer_pre  = TfidfVectorizer(stop_words = pt_stop_words, preprocessor=cleaner)


# Fit the vectorizer to labeled data

vectorizer_pre.fit(data.Texto)


# buildig the UpWeightVectorizer based on a whitelist

whitelist = [] #  read from a csv after the first run white list


# print idf values
df_idf = pd.DataFrame(vectorizer_pre.idf_, index=vectorizer_pre.get_feature_names(),columns=["tf_idf_weights"])

# sort, log
top_50_IFIDF_pre = df_idf.sort_values(by=['tf_idf_weights'], ascending=False).head(50)

##############   bui
def _make_upweighter(vectorizer, terms, weight):
    P = vectorizer.idf_.shape
    M = np.ones(P)
    idxs = [vectorizer.vocabulary_[t] for t in whitelist]
    M[idxs] = weight
    return M

def vectorizer(vectorizer, upweighter, doc):
    return vectorizer.transform(doc).multiply(upweighter)

up_weighter_vector_learned = _make_upweighter(vectorizer_pre, whitelist, 3)


from functools import partial
vectorizer_pos_UPweightining = partial(vectorizer, vectorizer_pre, up_weighter_vector_learned)

'''# prepare a sample from unknown data to compared to labeled dataset clustering properties
 A) only 5 % of unknown obs B)  only labeled  C)complete labeled add up to a sample of unknown
'''

import math
sample_A = random.sample(path_unknown,math.ceil(len(path_unknown)/20))
sample_B = path_evidence + path_not_evidence


''' Transforming the documents into vectors'''
''' Sample A'''

sample_Avect = [vectorizer_pos_UPweightining([open(path).read()]) for path in sample_A]

'''stack the list of sparse vectors into a sparse matrix'''
from scipy import sparse
sample_Avect = sparse.vstack(sample_Avect)

'''define label with unknown data = 2 for cluster analysis'''
y_unk = pd.Series(list (itertools.repeat(2, len(sample_A))))

''' Sample B'''
sample_Bvect = [vectorizer_pos_UPweightining([open(path).read()]) for path in sample_B]

''' stack the list of sparse vectors into a mantrix'''
from scipy import sparse
sample_Bvect = sparse.vstack(sample_Bvect)

'''sample c'''
sample_Cvect = sparse.vstack([sample_Bvect, sample_Avect])
sample_Avect.shape
sample_Bvect.shape
sample_Cvect.shape
#######################

''' Logging results'''

log.append("Size of the sparse matrix computed from all data: {} bytes".format(sys.getsizeof(sample_Cvect)))
log.append("Shape of the sparse matrix computed from all data: {}".format(sample_Cvect.shape))




'''############ This vectorizer takes raw documents instead of single documets '''

class UpWeightVectorizer():
    def __init__(self, whitelist, weight, **kwargs):
        self.whitelist = whitelist
        self.weight = weight
        self.vectorizer = TfidfVectorizer(**kwargs)


    def fit(self, X):
        self.vectorizer.fit(X)
        self.M = _make_upweighter(self.vectorizer, self.whitelist, self.weight)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X).multiply(self.M)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


######################################
''' Fitting the upweighter TFIDF object with the same docs than the partial fucntion. '''


Up_TFIDF_vectorizer_obj = UpWeightVectorizer(whitelist=whitelist, weight=3, preprocessor=cleaner, stop_words=pt_stop_words)

Up_TFIDF_vectorizer_obj.fit(data.Texto)

if Up_TFIDF_vectorizer_obj.vectorizer.idf_.shape == vectorizer_pre.idf_.shape:
    log.append("Both vectorizer's vocabulary have the same lenght")
else:
    log.append("WARNING. The shape of vectorizers vocabulary are not compatile. Object: {} and partial fuction {}.".format(Up_TFIDF_vectorizer_obj.vectorizer.idf_.shape , vectorizer_pre.idf_.shape))

vector_all =  [Up_TFIDF_vectorizer_obj.transform([open(path).read()]) for path in complete_path]

vector_all = sparse.vstack(vector_all)
log.append("Shape of sparse vector to all observatios: {}.".format(vector_all.shape))


######################################


# dimension reduction  and viz

def plot_embedding(V, y, title="Graphical representation of cosine similarity", given_hue='label'):
    """ Visualizes a vocabulary embedding via TSNE """
    V = TruncatedSVD(50).fit_transform(V)
    d = TSNE(metric='cosine').fit_transform(V)
    d = pd.DataFrame(d).assign(label = y.reset_index(drop=True))
    d['label'] = d['label'].replace([0,1,2],["Not evidence","Evidence","Not labeled"])
    ax = sns.scatterplot(x = 0, y = 1, hue = given_hue, data = d).set_title(title)
    now = date.today()
    ax.figure.savefig("Figure_{}_{}.png".format(title, now), dpi=300)
    plt.close(ax.figure)
    return ax, d,



first_plot, _ = plot_embedding(sample_Avect, y_unk, title= "Compare similarity: sample of unknown observations", given_hue=None)
second_plot, _ = plot_embedding(sample_Bvect,y, "Compare similarity: only labeled data")


# preparing the labels to plot labeled and the sampled unlabeled data together
# using for DBScan also

y_c =pd.concat((y,y_unk))
sample_Cvect.shape
len(y_c)

third_plot, _ = plot_embedding(sample_Cvect, y_c, "Compare similarity: labeled and a sample of unlabeled data")
plt.close('all')


#######################################################
''' Now we perform a clustering analysis based on module DBSCAN - Density-Based Spatial Clustering
Source: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html'''
#######################################################

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# #############################################################################
'''The function below export the main stats to a log file,
save a figure of clusters and export the object predictor further comparison '''
# #############################################################################



def compute_export_clusters (X, y, context="", export_predictor = False ):

    ''' Takes a sparse matrix, a vector of targets to process DBSCAN.
    The context variable is for the report. The predicator object will be kept to further analys on the SVM classifier.
    The sparse vector from TFIDF will have a dimension reduction to pass to standard DBScan implementation from Sklearn'''


    low_dim = TruncatedSVD(50).fit_transform(X)
    X_scan = StandardScaler().fit_transform(low_dim)

    labels_true = y
    db = DBSCAN(eps=0.3, min_samples=10).fit(low_dim)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
# append results to log
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    log.append('############## Start log results from DBSCAN {}'.format(context))
    log.append('Estimated number of clusters: %d' % n_clusters_)
    log.append('Estimated number of noise points: %d' % n_noise_)
    log.append("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    log.append("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    log.append("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    log.append("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
    log.append("Adjusted Mutual Information: %0.2f"
      % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
    log.append("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(low_dim, labels))

# ###
# Plot result
# Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)

        xy = low_dim[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

        xy = low_dim[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters {}: %d'.format(context) % n_clusters_)
    plt.savefig("Estimated clusters.png")
    ### Here we export the predictor object to compare with  SVM predictions
    ### Sampling from different cluster is a possible approach to construct upper confidence bounds

    if export_predictor == True:
        return db
    plt.close('all')
    pass

#######################################################
'''Call cluster analisys for diferent samples of documents '''
#######################################################

''' In the fist call we instantiate also the clustering predictor to use later'''

cluster_predictor = compute_export_clusters(sample_Cvect, y_c, "on a sample of unlabeled data and all labeled data.", export_predictor = True)
#compute_export_clusters(sample_Avect, y_unk, "on a sample of unlabeled data.")
compute_export_clusters(sample_Bvect, y, "on all the labeled data.")

'''
#####################################################
Get most important words of different cluster on not evidence.
The idea is to discard they from the analysis. TODO
'''


'''
#######################################################
'''



'''
Eplaining a random forest classifier will be our objetive.
A SGD with hinge loss and Gaussian Kernel transformation will our benchmarking classifier.

Here we already have the vectorizer_pre and the vectorizer objects created.
We setup a SGDClassifier with hinge loss because SVM defalut implementation has a quadratic complexity.
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
The module Kernel aproximation completes the strategy of building a scalable SVM classifier aproximation.
https://scikit-learn.org/stable/modules/kernel_approximation.html

'''
#
#######################################################

# instanciate SGD with hinge loss print_function

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(sample_Bvect)
X_features.shape

clf = SGDClassifier(max_iter=100, tol=1e-3, n_jobs=-1, loss='log')
clf.fit(X_features, y)

 #instanciate the RandomForestClassifier for benchmarking

clf_forest = RandomForestClassifier(n_estimators=100, class_weight="balanced")

'''
Prepare the data for the classifiers. Now using preparing pandas Data frame, not reading from disk, to use in LIME.

'''
#######################################################


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# compute class imbalance

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y), y)


now = datetime.now()
log.append("Results log. Run in {}.".format(now))
log.append("Class imbalance: {}".format(class_weights))
log.append("Labeled data shape: {}".format(data.shape))


'''################# f1 score  // recall analysis and vizualization'''

from sklearn.metrics import classification_report
X_train_vect = rbf_feature.fit_transform(vectorizer_pos_UPweightining(X_train))
X_test_vect = rbf_feature.fit_transform(vectorizer_pos_UPweightining(X_test))


log.append("Size of the sparse matrix learned from labeled data: {} bytes".format(1.25*(sys.getsizeof(X_train_vect))))
log.append("Shape of the sparse matrix learned from labeled data: {}".format(X_train_vect.shape))

def train_export_metrics (model, labels = ["Irrelevante","Evidência"]):
    log.append("##### Results from model: {}".format(model))
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    log.append(metrics.classification_report(y_test, y_pred))
    a = classification_report(y_test, y_pred, target_names=labels)
    log.append("{}".format(a))
    pass

# Run the results export

train_export_metrics(clf)
train_export_metrics(clf_forest)
'''
#######################################################

Prepare LIME to explain the predictions.



20 optimaly picked explanations
20 labeled as Evidence
20 best scored predictions over unlabeld data


#######################################################'''

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick



def export_explanations (model, X_test = X_test, class_names = ["Irrelevante", "Evidência"], name="name", n_exp=20):
    '''
    Arguments here are:
        Model has to be a pipeline with (vectorizer +  model) to explain.
        X_test = the subset of testing data in order to compute the best coverage of features to explain
        class names
        name is the argument that will name the files in disk
        and number of variables to pick
    '''
    class_names = ["Irrelevante","Evidência"]
    explainer = LimeTextExplainer(class_names=class_names)


        ######   Here begin the sub-modular pick code
    sp_obj = submodular_pick.SubmodularPick(explainer, X_test, model.predict_proba, sample_size=n_exp, num_features=15,num_exps_desired=n_exp)
    imagens = [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]
    i =0
    for exp in sp_obj.sp_explanations:
        exp.save_to_file(file_path="{}explanation{}.html".format(name,i))
        i+=1
    i=0
    for img in imagens:
        img.savefig("{}Imagem{}".format(name,i))
        i+=1
        #plt.close(img)
        plt.close('all')
    pass


'''  Alternative way to explain documents based on indexes. Delete if not usefull if compared to explain_any_list_doc

def explain_any_labeled (list_of_indexes, model, name="Labeled data", X = X):

    Arguments are 1) list of indexes to subset X (labeled data loaded previously as a Data Frame) and the pipe-model to explain

    class_names = ["Irrelevante","Evidência"]
    explainer = LimeTextExplainer(class_names=class_names)
    exps = [explainer.explain_instance(X[idx], model.predict_proba, num_features=10) for idx in list_of_indexes]
    images = [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in exps]
    for i, exp in enumerate(exps):
        exp.save_to_file(file_path="{}explanation{}.html".format(name,i))
    for i,img in enumerate(images):
        img.savefig("{}Imagem{}".format(name,i))
    pass
'''

def explain_any_list_doc (list_of_docs, model, name="Unlabeled"):
    '''
    Arguments are 1) list of documents to
    '''
    class_names = ["Irrelevante","Evidência"]
    explainer = LimeTextExplainer(class_names=class_names)
    exps = [explainer.explain_instance(doc, model.predict_proba, num_features=10) for doc in list_of_docs]
    images = [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in exps]
    for i, exp in enumerate(exps):
        exp.save_to_file(file_path="{}explanation{}.html".format(name,i))
    for i,img in enumerate(images):
        img.savefig("{}Imagem{}".format(name,i))
        plt.close(img)

    pass


'''
########################################################

Create pipeline objects to pass through LIME explainer

 Using SGD Classifier only as a benchmarking. The explanations will be performerd on random forest predictions
########################################################
'''

pipe_SGD_Up_vectorized = make_pipeline(Up_TFIDF_vectorizer_obj, rbf_feature, clf)
pipe_forest_Up_vectorized = make_pipeline(Up_TFIDF_vectorizer_obj, rbf_feature, clf_forest)
pipe_SGD_vanilla_vectorized = make_pipeline(vectorizer_pre, rbf_feature, clf)
pipe_forest_vanilla_vectorized = make_pipeline(vectorizer_pre, rbf_feature, clf_forest)


''' ##### Build a random list of evidences and explain it. We picked 10 '''
z = int(len(y[y==1]))
list_of_indexes = random.sample(list(range(z)), 10)
list_of_evidences = [X[i] for i in list_of_indexes]


explain_any_list_doc(list_of_evidences, pipe_forest_Up_vectorized, name="Evidências conhecidas")


''' Export optimized pick of evidence '''

export_explanations( pipe_SGD_Up_vectorized, X_test = X_test, name="SGD")
# export_explanations( pipe_SGD_Up_vectorized, X_test = X_test, name="Forest")

''' #### Predict labels for unlabeled data for both fitted model and pick the most likely to be evidence '''
''' prepere vectors '''

vect_unknown = [vectorizer_pos_UPweightining([open(path).read()]) for path in path_unknown]
vect_unknown = sparse.vstack(vect_unknown)
vect_unknown = rbf_feature.fit_transform(vect_unknown)

'''run predict_proba'''
preds_SGD = pipe_SGD_Up_vectorized.predict_proba(vect_unknown)
preds_forest = pipe_forest_Up_vectorized.predict_proba(vect_unknown)

'''
# sort and pick the more
# take a look at predictions distribution
'''
plt.figure()
a = sns.distplot(preds_SGD[:,1], kde=False, rug= True).set_title("Predictions distribution from  SGD Classifier.")
a = a.get_figure()
a.savefig('Predictions distribution from SGD classifier.png')
plt.close(a.get_figure())

'''
########################################################

 compute 20 for each classifier, merge, and get the unique values

########################################################
'''
idx_preds_SGD = preds_SGD[:,1].argsort()[-20:]
idx_preds_forest = preds_forest[:,1].argsort()[-20:]

idxs = np.concatenate((idx_preds_SGD , idx_preds_forest))
idx_preds = np.unique(idxs)
log.append("SGD and Forest best 20 predictions have {} elements in common. Indexes to explain:{}".format(40-len(idxs), idxs))

'''Load the files'''
files = [open(path_unknown[idx]).read() for idx in idx_preds]

explain_any_list_doc(files, pipe_forest_Up_vectorized, name=" Docs_higher_prob")


'''
Comparing correlation between:
1) clusters and classes on labeled data
2) predicted clusters and predicted classes on labeled data
3) Further research:  how to pick next set of document review using upper confidence bounds (Thomson's Sampling)

'''

clusters = cluster_predictor.fit_predict(vect_unknown)

sns.set_context("paper")
sns.set_style("white", {"axes.facecolor": ".9"})

sns.distplot(clusters).set_title("Number of observations in each cluster").get_figure().savefig("Number_of_obs_per_cluster.png")
plt.close('all')

for cluster in np.unique(cluster_predictor.labels_):
    a = sns.distplot(preds_SGD[:,1][clusters==cluster],kde=False, rug= True)
    a.set_title("Predictions distribution from  SGD Classifier in cluster {}.".format(cluster))
    a.get_figure().savefig("Predictions_cluster{}.png".format(cluster))
    a.get_figure()
    plt.close(a.get_figure())





sns.distplot(clusters).set_title("Number of observations in each cluster").get_figure().savefig("Number_of_obs_per_cluster.png")
plt.close('all')

np.savetxt("cluster_predictions.csv", clusters, delimiter=",")

##########
end_time = (time.time() -start_time)/60
log.append("This code runs in {:.2f} minutes".format(end_time))
##################
# export Results

def log_results(filename, results):
    with open(filename, 'a') as f:
        for r in results:
            f.write(f'{r}\n')
        f.write('\n')

log_results('log{}.txt'.format(now), log )
