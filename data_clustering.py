#!/usr/bin/env python2
# Based on algorithms from:
# The Data Science Lab: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# Scikit-learn: http://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine

def import_from_csv(file_name):
	print("Loading file: %s" % str(file_name))
	dataset=pd.read_csv(file_name, sep='\t')
	print("Loading finished.")
	return dataset

def export_to_csv(file_name,data):
	print("Exporting to file: %s" % str(file_name))
    dataset = pd.DataFrame.from_dict(data, orient='index')
    dataset.to_csv(file_name, sep='\t', na_rep='0.0')
	print("Exporting finished.")
	return

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

print"Program starts..."
dataset=import_from_csv(sys.argv[1])
X_train=init_vector(dataset)
centers, clusters = find_centers(X_train,5) # In this example K=5

print"Program ends..."