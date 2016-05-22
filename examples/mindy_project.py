 df=pd.DataFrame(data,index=['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'])
    df = pd.DataFrame(df)
    df.to_csv('clustering_KMeans.csv', sep='\t', encoding='utf-8')