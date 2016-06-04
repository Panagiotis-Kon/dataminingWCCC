from pandas import *
import string
import pandas as pd
import os
from Evaluation.EvaluationMetrics import classification,myMethod,getTextFeatures

df=pd.read_csv("train_set.csv",sep="\t")

# print("="*80)
# print("Classification with LDA features")
# for K in [10,100,1000]:
#     print("-"*80)
#     print "K:"+str(K)
#     X,X_LSI,Y=getTextFeatures(df,0,K)
#     d=classification(X,X_LSI,Y,K)
#     d=myMethod(d,X,Y,K)
#     dp = pd.DataFrame(d)
#     if not os.path.isfile('EvaluationMetric_10fold_lda_only.csv'):
#         dp.to_csv('EvaluationMetric_10fold_lda_only.csv', sep='\t', encoding='utf-8')
#     else:
#         dp.to_csv('EvaluationMetric_10fold_lda_only.csv', sep='\t', encoding='utf-8',mode = 'a',header=False)

print("="*80)
print("Classification with all features")
for K in [10,100,1000]:
    print("-"*80)
    print "K:"+str(K)
    X,X_LSI,Y=getTextFeatures(df,1,K)
    d=classification(X,X_LSI,Y,K)
    d=myMethod(d,X,Y,K)
    dp = pd.DataFrame(d)
    if not os.path.isfile('EvaluationMetric_10fold_ex1_features.csv'):
        dp.to_csv('EvaluationMetric_10fold_ex1_features.csv', sep='\t', encoding='utf-8')
    else:
        dp.to_csv('EvaluationMetric_10fold_ex1_features.csv', sep='\t', encoding='utf-8',mode = 'a',header=False)

# print("="*80)
# print("Test set prediction")
# df_test=pd.read_csv("test_set.csv",sep="\t")
# #testSetPrediction(df,df_test)
