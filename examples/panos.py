def execution(name,classifier):
    print("Training: ")
    print(classifier)

    if(name == "Multinomial Naive Bayes" or name == "Binomial Naive Bayes"):
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

    tuned_parameters={'svd__n_components':[50],'tfidf__use_idf':(True,False)}
    clf = GridSearchCV(pipeline, {}, cv=10,n_jobs=-1)
    clf.fit(X_train,Y_train_true)
    print("Best parameters set found on development set:")
    print
    print(clf.best_params_)
    print
    predicted=clf.predict(X_test)
    #print(metrics.classification_report(df['Category'], le.inverse_transform(predicted)))
    print(metrics.classification_report(Y_test_true,predicted,target_names=le.classes_))
	
	
	
df=pd.read_csv("train_set.csv",sep="\t")
X=df[['Title','Content']]
X_init=df[['Title','Content','Category']]
le=preprocessing.LabelEncoder()
le.fit(df["Category"])			# vectorise
Y=le.transform(df["Category"])	# enumeration

X_train, X_test, Y_train_true, Y_test_true = train_test_split(
    X, Y, test_size=0.2, random_state=0)

vectorizer=CountVectorizer(stop_words='english') # metra lekseis
transformer=TfidfTransformer()	# frequency count
svd=TruncatedSVD(n_components=10, random_state=42) # LSI, dimensionality n_components


X=vectorizer.fit_transform(X)
X=transformer.fit_transform(X)
X=svd.fit_transform(X)



for clf, name in (
        (BernoulliNB(alpha=0.2),"Binomial Naive Bayes"),
        (MultinomialNB(alpha=0.2),"Multinomial Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=10,n_jobs=-1), "k-Nearest Neighbor"),
        (SGDClassifier(n_jobs=-1), "SVM"),
        (RandomForestClassifier(n_estimators=100,n_jobs=-1), "Random forest")):
    print(name)
    execution(name,clf)