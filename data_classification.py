# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import pandas as pd

def data_classifier(classifier_name,classifier):
  print("Training: ")
  print(classifier)

  if(name == "Multinomial Naive Bayes" or name == "Binomial Naive Bayes"):
    pipeline = Pipeline([
      ('vect', vectorizer),
      ('tfidf', transformer),
      ('clf', classifier)
    ])
  else:
    pipeline = Pipeline([
      ('vect', vectorizer),
      ('tfidf', transformer),
      ('svd',svd),
      ('clf', classifier)
    ])

  tuned_parameters={'svd__n_components':[50],'tfidf__use_idf':(True,False)}
  clf = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
  clf.fit(X_train,Y_train_true)
  print("Best parameters set found on development set:")
  print
  print(clf.best_params_)
  print
  predicted=clf.predict(X_test)
  #print(metrics.classification_report(df['Category'], le.inverse_transform(predicted)))
  print(metrics.classification_report(Y_test_true,predicted,target_names=le.classes_))


print"Program starts..."
file_name=str(sys.argv[1])
df=pd.read_csv(file_name,sep="\t")
X=df[['Title','Content']]
X_init=df[['Title','Content','Category']]
le=preprocessing.LabelEncoder()
le.fit(df["Category"])      # vectorise
Y=le.transform(df["Category"])  # enumeration

X_train, X_test, Y_train_true, Y_test_true = train_test_split(
    X, Y, test_size=0.2, random_state=0)

vectorizer=CountVectorizer(stop_words='english') # metra lekseis
transformer=TfidfTransformer()  # frequency count
svd=TruncatedSVD(n_components=10, random_state=42) # LSI, dimensionality n_components


X=vectorizer.fit_transform(X)
X=transformer.fit_transform(X)
X=svd.fit_transform(X)



for clf, name in (
        (BernoulliNB(alpha=0.2),"Binomial Naive Bayes"),
        (MultinomialNB(alpha=0.2),"Multinomial Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=10,n_jobs=-1), "k-Nearest Neighbor"),
        (SGDClassifier(n_jobs=-1), "SVM"),
        (RandomForestClassifier(n_estimators=100,n_jobs=-1), "Random forest")):
    print(name)
    execution(name,clf)