import  pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


modelling  = DecisionTreeRegressor()
data = pd.read_csv('train.csv')
print len(data)
#print (data)

out = data.output
X = data[['X1','X2','X3','X4','X5','X6','X7','X8']]

train_x, train_y, val_x, val_y = train_test_split(X, out, random_state = 0)
print len(train_x)
print np.shape(train_x)
print len(train_y)
print np.shape(train_y)	
print len(val_x)
print np.shape(val_x)
print len(val_y)
print np.shape(val_y)
modelling.fit(train_x, val_x)
accuracy = modelling.score(train_y, val_y)
print accuracy



