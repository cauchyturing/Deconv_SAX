# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:04:45 2016

@author: Stephen-Lu
"""
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm
import bop
import numpy as np
from sklearn import cross_validation

def bop_vec(dataset, windowWidth, skipSize = 1, numSymbols = 3, alphabetSize = 7):

    features = {}
    Bops = []
    # Find all unique features and class labels
    for datum in dataset:
        bops = bop.sax_words(bop.standardize(datum),
                             windowWidth, skipSize, numSymbols, None, alphabetSize, True)
        bops = bop.bop(bops)
        for key in bops.keys():
            features[key] = 1
        Bops.append(bops)
#    if len(features) > 20000:
#      return
    # Stores features and labels as sorted lists
    features = sorted(features.keys())
    
    # Write data instances
    mat = []
    for bops in Bops:
        vec = []
        for feature in features:
            if feature in bops:
                vec.append(bops[feature])
            else:
                vec.append(0)
        mat.append(vec)
    return np.asarray(mat, dtype = np.float32)

    
npzfile = np.load('feature_train_ECG_MTS.npz')
feature_train = npzfile['x']
label_train = npzfile['y']

npzfile = np.load('feature_test_ECG_MTS.npz')
feature_test = npzfile['x']
label_test = npzfile['y']
train_num = feature_train.shape[0]
windowWidth = 0.05 #0.01
numSymbols = 3  #12
alphabetSize = 12 #7
feature = np.vstack((feature_train, feature_test))
bops = bop_vec(feature, windowWidth = int(windowWidth * feature.shape[1]), numSymbols = numSymbols, alphabetSize= alphabetSize )
bop_train = bops[:train_num]
bop_test = bops[train_num:]
#%%
clf = knn(n_neighbors=1)
clf.fit(bop_train, label_train)
print clf.score(bop_train, label_train)
print clf.score(bop_test, label_test)
#%%
#%%
clf = svm.LinearSVC()
clf.fit(bop_train, label_train)
svctrain = cross_validation.cross_val_score(clf, bop_train, label_train, cv=len(label_train)*2/3)
svctrain = np.mean(svctrain)
svctest = clf.score(bop_test, label_test)
print clf.score(bop_train, label_train)
print clf.score(bop_test, label_test)
print svctrain, svctest
#%%
clf = knn(n_neighbors=3)
clf.fit(bop_train, label_train)
print clf.score(bop_train, label_train)
print clf.score(bop_test, label_test)