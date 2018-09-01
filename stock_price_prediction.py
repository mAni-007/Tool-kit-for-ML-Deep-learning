import pandas as pd 
import quandl
import math

import arrow
import time
from datetime import datetime
import matplotlib.pyplot as plt 
from matplotlib import style

import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')



quandl.ApiConfig.api_key = 'ibDyF9TBEjBAAyQ6VgL8'
df = quandl.get('WIKI/GOOGL')
"""df  = open('data.txt','r')
df = df.readlines()"""

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head())
forecast_col = 'Adj. Close'
#print df
#to fill NAN places with -99999
df.fillna(-99999, inplace=True)



forecast_out = int(math.ceil(0.01*len(df)))

print df 

#adding label to dataframe
df['label'] = df[forecast_col].shift(-forecast_out)
print df 


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
print "value of X "
print np.shape(X)
print X


X_lately = X[-forecast_out:]
print "value of X after forecast_out"
print np.shape(X_lately)
print X_lately


X = X[:-forecast_out]
print "X after -forecast_oout"
print np.shape(X)
print X


df.dropna(inplace = True)
Y = np.array(df['label'])
print np.shape(Y)


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)
clf = LinearRegression()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)


forecast_set = clf.predict(X_lately)

print (forecast_set, accuracy, forecast_out)

"""for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)
    print(k,confidence)"""

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = arrow.get(last_date).timestamp
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = arrow.get(next_unix).timestamp
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()