from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import gensim
from gensim.models import LdaModel
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim import matutils
import numpy as np


#Read the data
df=pd.read_csv("train_set.csv",sep="\t")
#Preprocess the target variable (ex.1)
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])
X_train=df['Content']
#Use only he first 10K documents for this minimum example
documents = X_train[0:10000]
y=list(Y_train[0:10000])


#Convert docs to a list where elements are a tokens list
docsList=[document.lower().split() for document in documents]
#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
dictionary = corpora.Dictionary(docsList)
#Create the Gen-Sim corpus using the vectorizer
corpus = [dictionary.doc2bow(text.split()) for text in documents]
print("Preprocessing complete\n")
print("Training LDA\n")
#Train LDA with K=10
lda = LdaModel(corpus, id2word=dictionary, num_topics=10)
#For every doc get its topic distribution
corpus_lda = lda[corpus]
i=0
X=[]
print("Creating vectors")
#Create numpy arrays from the GenSim output
for l,t in zip(corpus_lda,corpus):
  ldaFeatures=np.zeros(10)
  for l_k in l:
  	ldaFeatures[l_k[0]]=l_k[1]
  X.append(ldaFeatures)
#Train the classifier ...
