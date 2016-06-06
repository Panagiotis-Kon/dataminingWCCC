from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pandas import *
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics  import *

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

import scipy.sparse as sparse
import re
import nltk
import sys
import string
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt

# stem_tokens is a help function for stemming the text
def stem_tokens(tokens, stemmer):

	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

#text_preprocessor will try to normalize the text. For doing so, it will
# convert the text to lowwercases only and tokenize the text
# at the end will stemm the words with lancaster stemmer
# For the preprocessing we used the nltk package
def text_preprocessor(text):

	lowers = text.lower()
	tokens = nltk.word_tokenize(lowers)
	lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
	stemmed_words = stem_tokens(tokens, lancaster_stemmer)
	return stemmed_words

# Lambda like function
def merger(x): 
	return x['Title']  + ' '+ x['Content']

# Tokenize the corpus with the help of nltk
# http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
def corpus_tokenizer(text):
	tokens = []
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	for t in text:
		lower = t.decode('utf-8').lower()
		token = tokenizer.tokenize(lower)
		tokens.append(token)
	return tokens

def LDA_processing(corpus, dictionary, k):

	lda = LdaModel(corpus, id2word=dictionary, num_topics=k)
	#For every doc get its topic distribution
	corpus_lda = lda[corpus]
	
	X_lda=[]
	print("Creating vectors...")
	#Create numpy arrays from the GenSim output
	for l,t in zip(corpus_lda,corpus):
  		ldaFeatures=np.zeros(k)
  		for l_k in l:
  			ldaFeatures[l_k[0]]=l_k[1]
  		X_lda.append(ldaFeatures)
  	print("Creating vectors finished...")
  	return X_lda

def Ex1_features(X):
	print("Vectorizer preprocessing starting...")
	vectorizer=CountVectorizer(stop_words='english')
	transformer=TfidfTransformer()
	svd=TruncatedSVD(n_components=20, random_state=42)
	X_vect=vectorizer.fit_transform(X)
	X_vect=transformer.fit_transform(X_vect)
	X_svd=sparse.csr_matrix(svd.fit_transform(X_vect))

	return X_vect,X_svd