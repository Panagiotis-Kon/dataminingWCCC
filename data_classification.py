

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
from scipy import spatial

from sklearn.metrics  import *

import string
import pandas as pd
from os import path
import matplotlib.pyplot as plt


# The classification function uses the pipeline in order to ease the procedure
# and also makes use of the GridSearchCV for the cross validation, without an tuned
# parameters, which makes it quicker
def classification(clfname,classifier):
    print('-' * 60)
    print("Training the Classifier: ")
    print(classifier)

    if(clfname == "(Binomial)-Naive Bayes" or clfname == "(Multinomial)-Naive Bayes"):

        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('clf', classifier)
        ])
        grid_search = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
        grid_search.fit(X_train,y_train)
    else:
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd',svd),
            ('clf', classifier)
        ])
        grid_search = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
        grid_search.fit(X_train,y_train)

    print
    print("Best params: " % grid_search.best_params_)
    print('*' * 60)
    predicted=grid_search.predict(X_test)
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_proba = grid_search.best_estimator_.predict_proba(X_test)
    y_test_boolean = []

    for index, item in enumerate(y_pred):
	    y_test_boolean.append(item == y_test[index])
    fpr, tpr, thresholds = metrics.roc_curve(y_test_boolean, y_proba[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    print ("Area under the ROC curve: %f" % roc_auc)

    accuracy = metrics.accuracy_score(y_test, predicted)
    print(metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predicted)))

    return fpr,tpr,roc_auc,accuracy

# Exports the validation results ton the csv
def export_to_csv():

    print("Exporting to csv...")
    dataframe = pd.DataFrame.from_dict(validation_results, orient='index')
    dataframe.to_csv("./data/EvaluationMetric_10fold.csv", sep='\t', encoding='utf-8', float_format='%.2f', index_label="Statistic Measure")


################################################################################
###################### Here starts the main of the program #####################
if __name__ == "__main__":

	print("Starting Classification Program")
	print ("#"*60)
	df=pd.read_csv("./data/train_set.csv",sep="\t")

    #do some preprocessing
	X=df[['Title','Content']]
	f=lambda x: x['Title']  + ' '+ x['Content']
	X=X.apply(f, 1)
	le=preprocessing.LabelEncoder()
	le.fit(df["Category"])
	y=le.transform(df["Category"])

    # split the train set (70 - 30) in order to have a small test set to check the classifiers

	X_train, X_test, y_train, y_test = train_test_split(
    	X, y, test_size=0.3, random_state=0)

	vectorizer=CountVectorizer(stop_words='english')
	transformer=TfidfTransformer()
	svd=TruncatedSVD(n_components=10, random_state=42)

    # initiate the array, which will hold all the results for the csv
	validation_results = {"Accuracy": {}, "ROC": {}}

	print("*"*60)
	print("Classification")

	classifiers_list = [(BernoulliNB(alpha=0.01),"(Binomial)-Naive Bayes","b"),
        	(MultinomialNB(alpha=0.01),"(Multinomial)-Naive Bayes","k"),
        	(KNeighborsClassifier(n_neighbors=9,n_jobs=-1), "k-Nearest Neighbor","r"),
        	(SVC(probability=True), "SVM","y"),
        	(RandomForestClassifier(n_estimators=1000,n_jobs=-1), "Random forest","g")]

	for clf, clfname, color in classifiers_list:
            print('=' * 60)
            print(clfname)
            fpr, tpr, roc_auc, accuracy_res = classification(clfname,clf)
            validation_results["Accuracy"][clfname] = accuracy_res
            validation_results["ROC"][clfname] = roc_auc
            plt.plot(fpr, tpr, 'k', label="%s ,Mean ROC (area = %0.2f)" % (clfname,roc_auc), lw=2, c="%s" % color)


    #create the ROC plot with the data generate from above
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc=2, fontsize='small')
	plt.savefig("./data/roc_10fold.png")

	export_to_csv()
