from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from pandas import *
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.cross_validation import train_test_split
from scipy import spatial
from nltk.stem import PorterStemmer
from sklearn.metrics  import *
from nltk.stem import WordNetLemmatizer
import nltk
import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
from gensim import corpora, models, similarities, matutils
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from pprint import pprint
import gensim
from numpy import concatenate
from scipy.sparse import csr_matrix
import scipy.sparse as sp

global X,Y,le,X_LSI

def classification(x,x_lsi,y,K):

    global X,Y,X_LSI
    X_LSI = x_lsi
    X = x
    Y = y
    d={}
    for clf, name in (
            (BernoulliNB(alpha=0.01),"Binomial Naive Bayes"),
            (MultinomialNB(alpha=0.01),"Multinomial Naive Bayes"),
            (KNeighborsClassifier(n_neighbors=9,n_jobs=-1), "k-Nearest Neighbor"),
            (svm.SVC(kernel='linear',probability = True), "SVM"),
            (RandomForestClassifier(n_estimators=1000,n_jobs=-1), "Random forest")):
        print('=' * 80)
        print(name)
        score=execution(name,clf)
        rec={name : pd.Series([score], index=['Accuracy K='+str(K)])}
        d.update(rec)
    return d

def getTextFeatures(df,mode,K):

    global X_LSI,X,Y,le

    X=df[['Title','Content']]
    X=X.apply(lambda x: x['Title']  + ' '+ x['Content'], 1)

    le=preprocessing.LabelEncoder()
    le.fit(df["Category"])
    Y=le.transform(df["Category"])

    if mode == 0:
        X=ldaFeatures(X,K)
        X_LSI=[]
    else:
        X1,X1_LSI=features(X)
        X2=ldaFeatures(X,K)
        X = sp.hstack((X1, X2), format='csr')
        X_LSI=[]
        #X=concatenate((X1,X2),axis=1)
        #X_LSI=sp.hstack((X1_LSI, X2), format='csr')
    return X,X_LSI,Y

def features(X):

    vectorizer=CountVectorizer(stop_words='english')
    transformer=TfidfTransformer()
    svd=TruncatedSVD(n_components=20, random_state=42)

    X=vectorizer.fit_transform(X)
    X=transformer.fit_transform(X)
    X_LSI=svd.fit_transform(X)

    #X = X.todense()
    return X,[]

def ldaFeatures(X,K):
    corpus = tokenize(X)
    id2word = corpora.Dictionary(corpus)
    corpus = [id2word.doc2bow(text) for text in corpus]

    lda = models.LdaModel(corpus,id2word=id2word ,alpha='auto',num_topics=K)
    lda_corpus = lda[corpus]


    X=[]
    for l,t in zip(lda_corpus,corpus):
      ldaFeatures=np.zeros(K)
      for l_k in l:
      	ldaFeatures[l_k[0]]=l_k[1]
      X.append(ldaFeatures)
    return X

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    tokens = []
    for i in text:
         raw = i.lower()
         t = tokenizer.tokenize(raw)
         stopped_tokens = [i for i in t if not i in en_stop]
         tokens.append(stopped_tokens)

    return tokens

def myMethod(d,X,Y,K):

    global X_train, X_test, Y_train_true, Y_test_true,le

    X_train, X_test, Y_train_true, Y_test_true = train_test_split(
        X, Y, test_size=0.3, random_state=0)

    clf=svm.LinearSVC(C=1)
    name="My Method"
    score=execution(name,clf)
    rec={name : pd.Series([score], index=['Accuracy K='+str(K)])}
    d.update(rec)
    return d


def testSetPrediction(df_train,df_test):

    X_train=df_train[['Title','Content']]
    X_train=X_train.apply(lambda x: x['Title']  + ' '+ x['Content'], 1)
    le=preprocessing.LabelEncoder()
    le.fit(df_train["Category"])
    Y_train_true=le.transform(df_train["Category"])

    X_Test=df_test[['Id','Title','Content']]
    X_init_test = X_Test
    X_Test=X_Test.apply(lambda x: x['Title']  + ' '+ x['Content'], 1)

    vectorizer=CountVectorizer(stop_words='english',tokenizer=tokenize_and_stem)
    transformer=TfidfTransformer()
    clf=svm.LinearSVC(C=1)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('clf', clf)
    ])
    clf = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
    clf.fit(X_train,Y_train_true)
    predicted=clf.predict(X_Test)


    id=[]
    for i in range(0,len(X_init_test)):
        id.append(X_init_test.ix[i][0])

    ID={'ID':id}
    cat=[]
    for p in predicted:
        cat.append(le.inverse_transform(p))
    Cat={'Predicted Category':cat}
    d = {}
    d.update(ID)
    d.update(Cat)
    p = pd.DataFrame(d)
    p.to_csv('testSet_categories.csv', sep='\t', encoding='utf-8')


def tokenize_and_stem(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = nltk.tokenize.word_tokenize(text)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def execution(name,clf):

    global X,Y,X_LSI
    print('_' * 80)
    print("Training: ")
    print(clf)

    if X_LSI != [] and name != "Multinomial Naive Bayes" and name!="Binomial Naive Bayes":
        X = X_LSI

    X_train, X_test, Y_train_true, Y_test_true = train_test_split(
        X, Y, test_size=0.3, random_state=0)

    clf = GridSearchCV(clf, {}, cv=10,n_jobs=-1)
    clf.fit(X_train,Y_train_true)
    predicted=clf.predict(X_test)
    print(metrics.classification_report(le.inverse_transform(Y_test_true), le.inverse_transform(predicted)))
    return accuracy_score(Y_test_true, predicted)
