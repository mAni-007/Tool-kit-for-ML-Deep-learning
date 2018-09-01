import pandas as pd 
import quandl
import math
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
#quandl.ApiConfig.api_key = '1et97dVuz3k5TWPAhQze'
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head())
forecast_col = 'Adj. Close'
#print df
#to fill NAN places with -99999
df.fillna(-99999, inplace=True)

#print df - df

forecast_out = int(math.ceil(0.01*len(df)))
#print forecast_out
#print  size(forecast_out)	
#print df.head()
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace = True)
#print df.head()



#X = np.array(df.drop(['label']), 1)
Y = np.array(df['label'])

#X = preprocessing.scale(X)
print Y
#X = X[:-forecast_out + 1]