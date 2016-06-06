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
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import scipy.sparse as sparse
import re
import nltk

import sys

from sklearn.metrics  import *

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

import string
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import data_csv_functions as dcsv

test_size=0.28
k_fold=10
k_neighbors_num=9
naive_bayes_a=0.05
svm_C=1.0
random_forests_estimators=100


# The classification function makes use of the GridSearchCV for the cross validation,
# without any tuned parameters, which makes it quicker

def default_classification(x,y,clfname,classifier):
	print('-' * 60)
	print("Default Classification Training %s" % clfname)
	# split the train set (75 - 25) in order to have a small test set to check the classifiers
	print("#"*60)
	print("Splitting the train set...")
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=test_size, random_state=0)
	print
	print(classifier)

	grid_search = GridSearchCV(classifier, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(x_train,y_train)
	print
	print('*' * 60)

	predicted=grid_search.predict(x_test)
	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy



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

def Lancaster_stemmer(text):
	stemmed = []
	lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
	for item in text:
		stemmed.append(lancaster_stemmer.stem(item))
	return stemmed

def MyMethod_classifier(x,y,clfname, classifier, user_input, k):
	print('-' * 60)
	print("My Method Classifier...")
	print
	
	print("LDA features for the My Method ")
	print

	vectorizer=CountVectorizer(stop_words='english',tokenizer=text_preprocessor)
    transformer=TfidfTransformer()

	#Convert docs to a list where elements are a tokens list
	corpus_train = corpus_tokenizer(x)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary_train = corpora.Dictionary(corpus_train)
	#Create the Gen-Sim corpus using the vectorizer
	corpus_train = [dictionary_train.doc2bow(text) for text in corpus_train]

	x_train_lda = LDA_processing(corpus_train, dictionary_train, k)

    print("Transforms...\n")
    x_train_vect=vectorizer.fit_transform(x)
	x_train_tfidf=transformer.fit_transform(x_train_vect)

	x_merged = None
	if user_input == 2:
		x_merged = sparse.hstack((x_train_tfidf, X_train_lda), format='csr')
		print("Merging of features finished")
		
	else:
		x_merged = x_train_lda

	# split the train set (75 - 25) in order to have a small test set to check the classifiers
	x_train, x_test, y_train, y_test = train_test_split(
		x_merged, y, test_size=test_size, random_state=0)
	

	print("GridSearchCV validation...")
	grid_search = GridSearchCV(classifier, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(x_train,y_train)
	print

	print('*' * 60)
	predicted=grid_search.predict(x_test)

	
	print("Calculate Metrics")
	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy

# predict_category: trains the whole dataset and makes predictions for the categories
# which are being exported to a csv file
def predict_category(X,y,k):

	print("Predict the category with SGD Classifier...")
	df_test = dcsv.import_from_csv(sys.argv[2])
	X_test_id = df_test[['Id','Title','Content']]
	X_test = X_test_id
	f=lambda x: x['Title']  + ' '+ x['Content']
	X_test = X_test.apply(f,1)

	
	X_train = X
	Y_train = y
	vectorizer = CountVectorizer(stop_words='english',tokenizer=text_preprocessor)
	transformer = TfidfTransformer()

	clf = SGDClassifier(loss='modified_huber',alpha=0.0001)

	###################### Preprocess the train set first ##################
	print("LDA features for the Train set")
	print
	#Convert docs to a list where elements are a tokens list
	corpus_train = corpus_tokenizer(X_train)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary_train = corpora.Dictionary(corpus_train)
	#Create the Gen-Sim corpus using the vectorizer
	corpus_train = [dictionary_train.doc2bow(text) for text in corpus_train]

	x_train_lda = LDA_processing(corpus_train, dictionary_train, k)

	print("Transforms in Train set")
	print
	x_train_vect=vectorizer.fit_transform(X_train)
	x_train_tfidf=transformer.fit_transform(x_train_vect)

	x_train_merged = sparse.hstack((x_train_tfidf, x_train_lda), format='csr')

	grid_search = GridSearchCV(clf, {}, cv=k_fold,n_jobs=-1)
	#Simple fit
	grid_search.fit(x_train_merged,Y_train)

	####################### TEST PREDICTION ###########################
	print("LDA features for the Test set")
	print
	#Convert docs to a list where elements are a tokens list
	corpus_test= corpus_tokenizer(X_test)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary_test = corpora.Dictionary(corpus_test)
	#Create the Gen-Sim corpus using the vectorizer
	corpus_test = [dictionary_test.doc2bow(text) for text in corpus_test]

	x_test_lda = LDA_processing(corpus_test, dictionary_test, k)

	print("Transforms in Test set")
	print
	x_test_vect=vectorizer.transform(X_test)
	x_test_tfidf=transformer.transform(x_test_vect)

	x_test_merged = sparse.hstack((x_test_tfidf, x_test_lda), format='csr')
	print("Starting the prediction")
	print
	#Predict the categories
	predicted=grid_search.predict(x_test_merged)

	# create lists to append the id from the test set
	# and the results from the prediction
	ID = []
	category = []
	
	for row in X_test_id.iterrows():
		index,data = row
		ID.append(data['Id'])
	id_dic = {'ID' : ID}

	for pred in predicted:
		category.append(le.inverse_transform(pred))
	category_dic = {'Predicted Category' : category}
	#finally append them to a dictionary for export
	out_dic = {}
	out_dic.update(id_dic)
	out_dic.update(category_dic)
	# Append the result to the csv
	print("Exporting predicted category to csv")
	dcsv.export_to_csv_categories("./data/testSet_categories.csv",out_dic)
	

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

################################################################################
###################### Here starts the main of the program #####################
if __name__ == "__main__":

	#print('sklearn version: {}.'.format(sklearn.__version__))
	print("Starting LDA Classification Program")
	print("#"*60)
	print("Give mode:\n")
	print("1: LDA features only")
	print("2: LDA features + ex1 features + Category prediction")
	print("0: exit")
	user_input = int(raw_input("Enter the number:  "))
	while (user_input!=0) and (user_input!=1) and (user_input!=2):
		user_input = int(raw_input("Enter the number again:  "))
	if (user_input==0):
		sys.exit()

	df=dcsv.import_from_csv(sys.argv[1])

	print("Preprocessing starting...\n")
	#merge content with title, in order to make use of the title help
	X=df[['Title','Content']]
	f=lambda x: x['Title']  + ' '+ x['Content']
	X=X.apply(f, 1)
	le=preprocessing.LabelEncoder()
	le.fit(df["Category"])
	y=le.transform(df["Category"])
	
	
	#Convert docs to a list where elements are a tokens list
	corpus = corpus_tokenizer(X)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary = corpora.Dictionary(corpus)
	#Create the Gen-Sim corpus using the vectorizer
	corpus = [dictionary.doc2bow(text) for text in corpus]
	print("Preprocessing complete...\n")

	combined_results = {"Accuracy K=10": {}, "Accuracy K=100": {}, "Accuracy K=1000": {}}
	lda_results = {"Accuracy K=10": {}, "Accuracy K=100": {}, "Accuracy K=1000": {}}

	# list of tuples for the classifiers
	# the tuple contains (classifier, name of the method, color for the auc plot)
	classifiers_list = [(BernoulliNB(alpha=naive_bayes_a),"(Binomial)-Naive Bayes"),
			(MultinomialNB(alpha=naive_bayes_a),"(Multinomial)-Naive Bayes"),
			(KNeighborsClassifier(n_neighbors=k_neighbors_num,n_jobs=-1), "k-Nearest Neighbor"),
			(SVC(probability=True), "SVM"),
			(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest"),
			(SGDClassifier(loss='modified_huber',alpha=0.0001), "My Method")]

	
	K=[100]
	for k in K:
		print("LDA Modeling starting...")
		print("   Number of Topics: %d \n" % k)
		

		X_lda = LDA_processing(corpus, dictionary, k)

		if user_input==2:
			X_vect,X_svd = Ex1_features(X)
			print("Vectorizer preprocessing finished...")
			
		print("*"*60)
		print("Classification starting...")

		#Loop through the classifiers list
		for clf, clfname in classifiers_list:
				print('=' * 60)
				print(clfname)
				accuracy_res = None
				if user_input==2:
					if clfname == "My Method":
						print("Combine LDA features + features...")
						accuracy_res = MyMethod_classifier(X, y, clfname, clf, user_input, k)
					else:
						if clfname == "(Binomial)-Naive Bayes" or clfname == "(Multinomial)-Naive Bayes":
							print("Combine LDA features + features...")
							X_merged = sparse.hstack((X_vect, X_lda), format='csr')
							accuracy_res = default_classification(X_merged, y, clfname, clf)
						else:
							print("Combine LDA features + features...")
							X_merged_svd = sparse.hstack((X_svd, X_lda), format='csr')
							accuracy_res = default_classification(X_merged_svd, y, clfname, clf)
						
					if k==10:
						combined_results["Accuracy K=10"][clfname] = accuracy_res
					elif k==100:
						combined_results["Accuracy K=100"][clfname] = accuracy_res
					else :
						combined_results["Accuracy K=1000"][clfname] = accuracy_res
				else:
					if clfname == "My Method":
						accuracy_res = MyMethod_classifier(X, y, clfname, clf, user_input, k)
					else:
						accuracy_res = default_classification(X_lda, y, clfname, clf)

					if k==10:
						lda_results["Accuracy K=10"][clfname] = accuracy_res
					elif k==100:
						lda_results["Accuracy K=100"][clfname] = accuracy_res
					else :
						lda_results["Accuracy K=1000"][clfname] = accuracy_res
		
	if user_input==2:
		print("LDA features + ex1 features")
		dcsv.export_to_csv_statistic("./data/EvaluationMetric_10fold_ex1_features.csv",combined_results)	
	else:			
		print("LDA features only")
		dcsv.export_to_csv_statistic("./data/EvaluationMetric_10fold_lda_only.csv",lda_results)

	predict_category(X,y,100)
	print("*"*60)
	print
	print("Program Exits....")
	print
	print("*"*60)
	exit()
