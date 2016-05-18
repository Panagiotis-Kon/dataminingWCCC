import pandas as pd

df=pd.read_csv("train_set.csv",sep='\t')
print(df.columns)
print(df["Category"])
#Conditional Selection
df.ix[df["Category"]=="Film"]
#Iteration example
for index, row in df.iterrows():
	print row["Category"], row["Id"]