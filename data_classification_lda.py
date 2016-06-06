from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.svm import SVC

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

import data_csv_functions as dcsv
import data_feature_functions as dff
import data_predict as dp

test_size=0.25
k_fold=10
k_neighbors_num=9
naive_bayes_a=0.05
svm_C=1.0
random_forests_estimators=100


# The default_classification function makes use of the GridSearchCV for the cross validation,
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

	print("Calculating Metrics...")
	print
	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy

# MyMethod_classifier function will do first some better preprocessing to our 
# train sample in order to achieve a better outcome
def MyMethod_classifier(x,y,clfname, classifier, user_input, k):
	print('-' * 60)
	print("My Method Classifier...")
	print
	
	print("LDA features for the My Method ")
	print

	vectorizer=CountVectorizer(stop_words='english',tokenizer=dff.text_preprocessor)
	transformer=TfidfTransformer()

	#Convert docs to a list where elements are a tokens list
	corpus_train = dff.corpus_tokenizer(x)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary_train = corpora.Dictionary(corpus_train)
	#Create the Gen-Sim corpus using the vectorizer
	corpus_train = [dictionary_train.doc2bow(text) for text in corpus_train]

	x_train_lda = dff.LDA_processing(corpus_train, dictionary_train, k)

	print("MyMethod_classifier transforms and merging...\n")
	x_train_vect=vectorizer.fit_transform(x)
	x_train_tfidf=transformer.fit_transform(x_train_vect)

	#Merge the features from lda and ex1 with the help of scipy package
	x_merged = None
	if user_input == 2:
		x_merged = sparse.hstack((x_train_tfidf, x_train_lda), format='csr')
		print("Merging of features finished")
		
	else:
		x_merged = x_train_lda

	# split the train set (75 - 25) in order to have a small test set to check the classifiers
	x_train, x_test, y_train, y_test = train_test_split(
		x_merged, y, test_size=test_size, random_state=0)
	

	grid_search = GridSearchCV(classifier, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(x_train,y_train)
	print

	print('*' * 60)
	predicted=grid_search.predict(x_test)

	
	print("Calculating Metrics...")
	print
	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy

################################################################################
###################### Here starts the main of the program #####################
if __name__ == "__main__":

	
	print("Starting LDA Classification Program")
	print("#"*60)
	print("Give mode:\n")
	print("1: LDA features only")
	print("2: LDA features + ex1 features")
	print("3: Category prediction")
	print("0: exit")
	user_input = int(raw_input("Enter the number:  "))
	while (user_input!=0) and (user_input!=1) and (user_input!=2) and (user_input!=3):
		user_input = int(raw_input("Enter the number again:  "))

	print
	if (user_input==0):
		print("Program exits...")
		sys.exit()
	elif (user_input==1):
		print("LDA features selected...")
	elif (user_input==2):
		print("LDA features + ex1 features selected...")
	else:
		print("Category prediction selected...")
	print
	print("#"*60)

	df=dcsv.import_from_csv(sys.argv[1])

	print("Preprocessing starting...\n")
	#merge content with title, in order to make use of the title help
	X=df[['Title','Content']]
	f=lambda x: x['Title']  + ' '+ x['Content']
	X=X.apply(f, 1)
	le=preprocessing.LabelEncoder()
	le.fit(df["Category"])
	y=le.transform(df["Category"])
	
	# if the user wants to make a prediction test
	if user_input==3:
		dp.predict_category(X,y,10,le,sys.argv[2])
		print("*"*60)
		print
		print("Program Exits....")
		print
		print("*"*60)
		sys.exit()

	
	# STARTING LDA FEATURES #
	#Convert docs to a list where elements are a tokens list
	corpus = dff.corpus_tokenizer(X)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary = corpora.Dictionary(corpus)
	#Create the Gen-Sim corpus using the vectorizer
	corpus = [dictionary.doc2bow(text) for text in corpus]
	print("Preprocessing complete...\n")

	# Initialize dictionaries for the results #
	combined_results = {"Accuracy K=10": {}, "Accuracy K=50": {}, "Accuracy K=100": {}, "Accuracy K=1000": {}}
	lda_results = {"Accuracy K=10": {}, "Accuracy K=50": {}, "Accuracy K=100": {}, "Accuracy K=1000": {}}

	# list of tuples for the classifiers
	# the tuple contains (classifier, name of the method, color for the auc plot)
	classifiers_list = [(BernoulliNB(alpha=naive_bayes_a),"(Binomial)-Naive Bayes"),
			(MultinomialNB(alpha=naive_bayes_a),"(Multinomial)-Naive Bayes"),
			(KNeighborsClassifier(n_neighbors=k_neighbors_num,n_jobs=-1), "k-Nearest Neighbor"),
			(SVC(probability=True), "SVM"),
			(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest"),
			(SGDClassifier(loss='modified_huber',alpha=0.0001), "My Method")]

	# Starting the main loop for the topics #
	K=[10]
	for k in K:
		print("LDA Modeling starting...")
		print("   Number of Topics: %d \n" % k)
		

		X_lda = dff.LDA_processing(corpus, dictionary, k)
		# if the user has given the option of exploiting lda features
		# along with the ex1 features...
		if user_input==2:
			# EX1 features...
			X_vect,X_svd = dff.Ex1_features(X)
			print("Ex1 features is finished...")
			
		print
		print("*"*60)
		print("Classification starting...")

		#Loop through the classifiers list
		for clf, clfname in classifiers_list:
				print('=' * 60)
				print(clfname)
				# for the Nave-Bayes classifiers we will not use the lsi features from svd

				accuracy_res = None
				if user_input==2:
					if clfname == "My Method":
						accuracy_res = MyMethod_classifier(X, y, clfname, clf, user_input, k)
					else:
						if clfname == "(Binomial)-Naive Bayes" or clfname == "(Multinomial)-Naive Bayes":
							print("Combine LDA features + Ex1 features...\n")
							X_merged = sparse.hstack((X_vect, X_lda), format='csr')
							accuracy_res = default_classification(X_merged, y, clfname, clf)
						else:
							print("Combine LDA features + Ex1 features...\n")
							X_merged_svd = sparse.hstack((X_svd, X_lda), format='csr')
							accuracy_res = default_classification(X_merged_svd, y, clfname, clf)
						
					if k==10:
						combined_results["Accuracy K=10"][clfname] = accuracy_res
					elif k==50:
						combined_results["Accuracy K=50"][clfname] = accuracy_res
					elif k==100:
						combined_results["Accuracy K=100"][clfname] = accuracy_res
					else :
						combined_results["Accuracy K=1000"][clfname] = accuracy_res
				else:
					# using only LDA features
					if clfname == "My Method":
						accuracy_res = MyMethod_classifier(X, y, clfname, clf, user_input, k)
					else:
						accuracy_res = default_classification(X_lda, y, clfname, clf)

					if k==10:
						lda_results["Accuracy K=10"][clfname] = accuracy_res
					elif k==50:
						lda_results["Accuracy K=50"][clfname] = accuracy_res
					elif k==100:
						lda_results["Accuracy K=100"][clfname] = accuracy_res
					else :
						lda_results["Accuracy K=1000"][clfname] = accuracy_res
	
	# Export the results accordingly...	
	if user_input==2:
		print("LDA features + ex1 features Export")
		dcsv.export_to_csv_statistic("./data/EvaluationMetric_10fold_ex1_features.csv",combined_results)	
	else:			
		print("LDA features only Export")
		dcsv.export_to_csv_statistic("./data/EvaluationMetric_10fold_lda_only.csv",lda_results)

	print("*"*60)
	print
	print("Program Exits....")
	print
	print("*"*60)
	sys.exit()
