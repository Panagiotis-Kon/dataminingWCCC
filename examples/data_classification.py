
  
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pandas import *
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.cross_validation import train_test_split
from scipy import spatial

from sklearn.metrics  import *

import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from sklearn import svm


def classification(name,classifier):
    print('-' * 80)
    print("Training: ")
    print(classifier)

    if(name == "BernoulliNB" or name == "MultinomialNB"):
      
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('clf', classifier)
        ])
        #tuned_parameters={'svd__n_components':[10,20,30,50],'tfidf__use_idf':(True,False)}
        clf = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
        clf.fit(X_train,Y_train)
    else:
          pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd',svd),
            ('clf', classifier)
          ])
        #tuned_parameters={'clf__alpha':[0.01,0.02,0.05,0.08,0.1]}
        clf = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
        clf.fit(X_train,Y_train)

    print
    print(clf.best_params_)
    print('*' * 80)
    predicted=clf.predict(X_test)
    print(metrics.classification_report(le.inverse_transform(Y_test), le.inverse_transform(predicted)))
    #print(metrics.classification_report(Y_test_true,predicted,target_names=le.classes_))


################################################################################
print("Starting Classification Program")
print ("#"*80)
df=pd.read_csv("train_set.csv",sep="\t")

X=df[['Title','Content']]
f=lambda x: x['Title']  + ' '+ x['Content']
X=X.apply(f, 1)
le=preprocessing.LabelEncoder()
le.fit(df["Category"])
Y=le.transform(df["Category"])


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer()
svd=TruncatedSVD(n_components=10, random_state=42)


print("*"*80)
print("Classification")
for clf, name in (
        (BernoulliNB(alpha=0.01),"Binomial Naive Bayes"),
        (MultinomialNB(alpha=0.01),"Multinomial Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=9,n_jobs=-1), "k-Nearest Neighbor"),
        (svm.SVC(), "SVM"),
        (RandomForestClassifier(n_estimators=1000,n_jobs=-1), "Random forest")):
    print('=' * 80)
    print(name)
    classification(name,clf)
    
