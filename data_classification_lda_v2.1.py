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
from scipy import spatial, interp
import re
import nltk
nltk.download()
import sys

from sklearn.metrics  import *

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import data_csv_functions as dcsv

test_size=0.25
k_fold=10
k_neighbors_num=9
naive_bayes_a=0.05
svm_C=1.0
random_forests_estimators=100


# The classification function uses the pipeline in order to ease the procedure
# and also makes use of the GridSearchCV for the cross validation, without any tuned
# parameters, which makes it quicker
def classification(clfname,classifier):
	print('-' * 60)
	print("Training %s" % clfname)
	print
	print(classifier)

	if(clfname == "(Binomial)-Naive Bayes" or clfname == "(Multinomial)-Naive Bayes"):

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

	grid_search = GridSearchCV(pipeline, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(X_train,y_train)
	print
	print('*' * 60)
	predicted=grid_search.predict(X_test)
	y_proba = grid_search.best_estimator_.predict_proba(X_test)

	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy,y_proba


def classification_lda(clfname,classifier,x_train,y_train,x_test,y_test):
	print('-' * 60)
	print("Training %s" % clfname)
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

# beat_the benchmark: this function tries to beat the other classifiers
# using the SGDClassifier, but doing some better preprocessing at the beginning
# Setting the loss parameter to modified_huber we generate a smoothed hinge loss.
# This allow not only to use the linear versio of the classifier but also enables
# predict_proba method, which gives a vector of probability estimates and help us
# to calculate the auc score.
def beat_the_benchmark(X,y,clfname,classifier):
	print('-' * 60)
	print("Beating the Benchmark...")
	print
	print("Preprocessing...")

	# split the train set (75 - 25) in order to have a small test set to check the classifiers
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=0)
	# CountVectorizer will call the function text_preprocessor
	vectorizer=CountVectorizer(stop_words='english',tokenizer=text_preprocessor)
	transformer=TfidfTransformer()

	print("Training %s" % clfname)
	print
	print(classifier)
	# In the pipeline we dont use the lsi because it will shrink valuable information
	# for the prediction. Moreover it's not so good to use it with
	# the SGDClassifier, which tunes the loss parameter at modified_huber and makes
	# it linear
	pipeline = Pipeline([
		('vect', vectorizer),
		('tfidf', transformer),
		('clf', classifier)
	])
	print("GridSearchCV validation...")
	grid_search = GridSearchCV(pipeline, {}, cv=k_fold,n_jobs=-1)
	grid_search.fit(X_train,y_train)
	print

	print('*' * 60)
	predicted=grid_search.predict(X_test)

	y_proba = grid_search.best_estimator_.predict_proba(X_test)
	print("Calculate Metrics")
	accuracy = metrics.accuracy_score(y_test, predicted)
	print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

	return accuracy,y_proba,y_test

# predict_category: trains the whole dataset and makes predictions for the categories
# which are being exported to a csv file
def predict_category(X,y,file_name):
	print("Predict the category with (Multinomial)-Naive Bayes Classifier...")
	X_train = X
	Y_train = y

	df_test = dcsv.import_from_csv(file_name)
	X_true_id = df["Id"]

	vectorizer=CountVectorizer(stop_words='english')
	transformer=TfidfTransformer()
	clf=MultinomialNB(alpha=naive_bayes_a)


	pipeline = Pipeline([
		('vect', vectorizer),
		('tfidf', transformer),
		('clf', clf)
	])
	#Simple Pipeline Fit
	pipeline.fit(X_train,Y_train)
	#Predict the train set
	predicted=pipeline.predict(X_train)
	# create lists to append the id from the test set
	# and the results from the prediction
	ID = []
	category = []
	for i in X_true_id:
		ID.append(i)
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


def merger(x):
	return x['Title']  + ' '+ x['Content']

def corpus_tokenizer(text):
	tokens = []
	#splitter = []
	#re.split(r'\W+',text)
	tokens = [nltk.word_tokenize(t.decode('utf-8').lower()) for t in text]
	return tokens
################################################################################
###################### Here starts the main of the program #####################
if __name__ == "__main__":

	print("Starting LDA Classification Program")
	print ("#"*60)
	df=dcsv.import_from_csv(sys.argv[1])

	#merge content with title, in order to make use of the title help
	X=df[['Title','Content']]
	#f=lambda x: x['Title']  + ' '+ x['Content']
	
	X=X.apply(merger, 1)
	le=preprocessing.LabelEncoder()
	le.fit(df["Category"])
	y=le.transform(df["Category"])
	


	#Convert docs to a list where elements are a tokens list
	#docsList=[document.lower().split() for document in documents]
	corpus = corpus_tokenizer(X)
	#Create Gen-Sim dictionary (Similar to SKLearn vectorizer)
	dictionary = corpora.Dictionary(corpus)
	#Create the Gen-Sim corpus using the vectorizer
	corpus = [dictionary.doc2bow(text) for text in corpus]
	print("Preprocessing complete\n")
	validation_results = {"Accuracy K=10": {}, "Accuracy K=100": {}, "Accuracy K=1000": {}}

	print("LDA preprocessing...")
	K=[10]
	for k in K:
		print("Number of Topics: %d \n" % k)
		
		lda = LdaModel(corpus, id2word=dictionary, num_topics=k)
		#For every doc get its topic distribution
		corpus_lda = lda[corpus]
		i=0
		X_lda=[]
		print("Creating vectors")
		#Create numpy arrays from the GenSim output
		for l,t in zip(corpus_lda,corpus):
  			ldaFeatures=np.zeros(k)
  			for l_k in l:
  				ldaFeatures[l_k[0]]=l_k[1]
  			X_lda.append(ldaFeatures)
		#Train the classifier ...

		# make a prediction for the category
		#predict_category(X,y,sys.argv[2])

		print("Vectorizer preprocessing...")
		vectorizer=CountVectorizer(stop_words='english')
		transformer=TfidfTransformer()
		svd=TruncatedSVD(n_components=20, random_state=42)
		X_vect=vectorizer.fit_transform(X)
		X_vect=transformer.fit_transform(X_vect)
		X_vect=svd.fit_transform(X_vect)

		print("Combine LDA + Vectorizer...")
		X_both = sparce.hstack((X_vect, X_lda), format='csr')

		# split the train set (75 - 25) in order to have a small test set to check the classifiers
		print("#"*60)
		print("Splitting the train set and doing some preprocessing...")
		x_train, x_test, y_train, y_test = train_test_split(
			X_both, y, test_size=test_size, random_state=0)



		# initiate the array, which will hold all the results for the csv
		

		print("*"*60)
		print("Classification")


		# list of tuples for the classifiers
		# the tuple contains (classifier, name of the method, color for the auc plot)
		classifiers_list = [(BernoulliNB(alpha=naive_bayes_a),"(Binomial)-Naive Bayes","b"),
				(MultinomialNB(alpha=naive_bayes_a),"(Multinomial)-Naive Bayes","k"),
				(KNeighborsClassifier(n_neighbors=k_neighbors_num,n_jobs=-1), "k-Nearest Neighbor","r"),
				(SVC(probability=True), "SVM","y"),
				(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest","g"),
				(SGDClassifier(loss='modified_huber',alpha=0.0001), "My Method", "m")]

		#Loop through the classifiers list. If it is the My Method then call beat the benchmark
		for clf, clfname, color in classifiers_list:
				print('=' * 60)
				print(clfname)
				accuracy_res = classification_lda(clfname,clf,x_train,y_train,x_test,y_test)
				if k==10:
					validation_results["Accuracy K=10"][clfname] = accuracy_res
				elif k==100:
					validation_results["Accuracy K=100"][clfname] = accuracy_res
				else :
					validation_results["Accuracy K=1000"][clfname] = accuracy_res


	

	dcsv.export_to_csv_statistic("./data/EvaluationMetric_10fold_lda_only.csv",validation_results)
