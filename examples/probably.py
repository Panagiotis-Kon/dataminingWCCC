    data=[]
    for cluster in clusters:
        politics=business=football=film=technology=0.0
        dataLength=len(cluster)
        for x in cluster:
            itemindex = np.where(x==X)
            if X_init.ix[itemindex[0][0]][2] == "Politics":
                politics+=1.0
            elif X_init.ix[itemindex[0][0]][2] == "Business":
                business+=1.0
            elif X_init.ix[itemindex[0][0]][2] == "Football":
                football+=1.0
            elif X_init.ix[itemindex[0][0]][2] == "Film":
                film+=1.0
            else:
                technology+=1.0
        d={'Politics':politics/dataLength,'Business':business/dataLength,'Football':football/dataLength,'Film':film/dataLength,'technology':technology/dataLength}
        data.append(d)
	df = pd.DataFrame(df)
    df.to_csv('clustering_KMeans.csv', sep='\t', encoding='utf-8')