

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