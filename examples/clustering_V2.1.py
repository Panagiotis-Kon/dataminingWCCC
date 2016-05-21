import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine

MAX_ITERATIONS = 50


# TODO review
def vectorization(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    x_tfidf = vectorizer.fit_transform(data['Content'])
    svd = TruncatedSVD(n_components=100)
    x_lsi = svd.fit_transform(x_tfidf)
    return x_lsi


# Based on algorithms found at
# 1) Stanford University's website: http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
# 2) 'The Data Science Lab' wordpress: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python
def clustering_k_means(x, k):
    centroids = random.sample(x, k)
    old_centroids = random.sample(x, k)
    iterations = 0
    clusters = {}
    while not should_stop(centroids, old_centroids, iterations):
        old_centroids = centroids
        iterations += 1
        clusters = get_labels(x, centroids)
        centroids = get_centroids(clusters)
    return centroids, clusters


def should_stop(centroids, old_centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    return set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids])


def get_labels(x, centroids):
    clusters = {}
    for node in x:
        center = min([(i, cosine(node, centroids[i])) for i in range(len(centroids))], key=lambda t: t[1])[0]
        try:
            clusters[center].append(node)
        except KeyError:
            clusters[center] = [node]
    return clusters


def get_centroids(clusters):
    centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        centroids.append(np.mean(clusters[k], axis=0))
    return centroids


def analyze_data(clusters, x_train, df):
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
    return results


def export_to_cvs(results):
    dataframe = pd.DataFrame.from_dict(results, orient='index')
    dataframe.to_csv("clustering_KMeans.csv", sep='\t', na_rep='0.0')


print ("Opening train dataset")
train_df = pd.read_csv('data/train_set.csv', sep='\t')
print ("Converting documents into vectors")
X_train = vectorization(train_df)
print ("Clustering based on k-means algorithm")
k_centers, k_cluster = clustering_k_means(X_train, 5)
print ("Analyzing clusters")
train_results = analyze_data(k_cluster, X_train, train_df)
print ("Exporting results")
export_to_cvs(train_results)
print ("Done")
