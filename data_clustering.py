#!/usr/bin/env python2
# Based on algorithms from:
# The Data Science Lab: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# Scikit-learn: http://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

import sys
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import data_csv_functions as dcvs

def init_vector(data):
	print"Vectorization of data initializes."
	vectorizer = TfidfVectorizer(stop_words='english')
	x_tfidf = vectorizer.fit_transform(data['Content'])
	svd = TruncatedSVD(n_components=100)
	x_lsi = svd.fit_transform(x_tfidf)
	print"Vectorization finished."
	return x_lsi

def cluster_points(X, mu): # Changed np.linalg.norm to cosine for this example
	clusters  = {}
	for x in X:
		bestmukey = min([(i[0], cosine(x,mu[i[0]])) \
					for i in enumerate(mu)], key=lambda t:t[1])[0]
		try:
			clusters[bestmukey].append(x)
		except KeyError:
			clusters[bestmukey] = [x]
	return clusters

def reevaluate_centers(mu, clusters):
	newmu = []
	keys = sorted(clusters.keys())
	for k in keys:
		newmu.append(np.mean(clusters[k], axis = 0))
	return newmu

def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
	# Initialize to K random centers
	print"K-means' clustering begins."
	sys.stdout.write("Processing: ")
	oldmu = random.sample(X, K)
	mu = random.sample(X, K)
	while not has_converged(mu, oldmu):
		# update the bar
		sys.stdout.write("#")
		sys.stdout.flush()
		oldmu = mu
		# Assign all points in X to clusters
		clusters = cluster_points(X, mu)
		# Reevaluate centers
		mu = reevaluate_centers(oldmu, clusters)
	print
	print"K-means' clustering finished."
	return (mu, clusters)

def generate_formated_results(dataset,X_train,clusters):
	print"Genarating results."
	sys.stdout.write("Processing: ")
	category_list = dataset['Category'].tolist()
	formated_results = {}
	for cluster, vectors in clusters.iteritems():
		# update the bar
		sys.stdout.write("#")
		sys.stdout.flush()
		total = 0
		cluster_index = "Cluster" + str(cluster + 1)
		formated_results[cluster_index] = {}
		for vector in vectors:
			total += 1
			category = category_list[np.where(X_train == vector)[0][0]]
			try:
				formated_results[cluster_index][category] += 1
			except KeyError:
				formated_results[cluster_index][category] = 1
		for category in formated_results[cluster_index]:
			formated_results[cluster_index][category] = round(formated_results[cluster_index][category] / float(total), 2)
			#formated_results[cluster_index][category] = formated_results[cluster_index][category] / float(total)
	print
	print"Genarating results finished."
	return formated_results

# The main of the program start here #
if __name__ == "__main__":
	print"Program starts..."
	print('=' * 60)
	dataset=dcvs.import_from_csv(sys.argv[1])
	X_train=init_vector(dataset)
	print('+' * 60)
	centers, clusters = find_centers(X_train,5) # In this example K=5
	print('-' * 60)
	formated_results=generate_formated_results(dataset, X_train, clusters)
	print('+' * 60)
	dcvs.export_to_csv_cluster('./data/clustering_KMeans.csv',formated_results)
	print('=' * 60)
	print"Program ends..."