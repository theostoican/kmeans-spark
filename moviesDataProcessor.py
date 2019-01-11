import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv("movies_metadata.csv")

df.drop(df.index[19730],inplace=True)
df.drop(df.index[29502],inplace=True)
df.drop(df.index[35585],inplace=True)

df_numeric = df[['budget','popularity','revenue','runtime','vote_average','vote_count','title']]

print df_numeric.head()
print df_numeric.isnull().sum()

df_numeric.dropna(inplace=True)

print df_numeric['vote_count'].describe()

df_numeric['vote_count'].quantile(np.arange(.74,1,0.01))
df_numeric = df_numeric[df_numeric['vote_count']>30]

minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title',axis=1))

df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])

df_numeric_scaled.to_csv("movies_processed.csv")