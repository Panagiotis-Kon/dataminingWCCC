import data_csv_functions as dcsv

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