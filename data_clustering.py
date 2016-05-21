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
from data_csv_functions import import_from_csv
from data_csv_functions import export_to_csv

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
	print"K-means' clusterization begins."
	oldmu = random.sample(X, K)
	mu = random.sample(X, K)
	while not has_converged(mu, oldmu):
		oldmu = mu
		# Assign all points in X to clusters
		clusters = cluster_points(X, mu)
		# Reevaluate centers
		mu = reevaluate_centers(oldmu, clusters)
	print"K-means' clusterization finished."
	return (mu, clusters)

def generate_results(clusters, x_train, df):
	print"Genarating results."
	ordered_list_with_category = df['Category'].tolist()
	results = {}
	for cluster, vectors in clusters.iteritems():
		total = 0
		cluster_name = "Cluster " + str(cluster + 1)
		results[cluster_name] = {}
		for vector in vectors:
			total += 1
			category = ordered_list_with_category[np.where(x_train == vector)[0][0]]
			try:
				results[cluster_name][category] += 1
			except KeyError:
				results[cluster_name][category] = 1
		for category in results[cluster_name]:
			 results[cluster_name][category] = round(results[cluster_name][category] / float(total), 2)
	print"Genarating results finished."
	return results

print"Program starts..."
dataset=import_from_csv(sys.argv[1])
X_train=init_vector(dataset)
centers, clusters = find_centers(X_train,5) # In this example K=5
results=generate_results(clusters, X_train, dataset)
export_to_csv('./data/clustering_KMeans.csv',results)
print"Program ends..."