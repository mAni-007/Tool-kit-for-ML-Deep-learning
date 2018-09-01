from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing  import OneHotEncoder
import csv
import numpy as np 
import pandas as pd 
data = pd.read_csv("dataset-fb-valence-arousal-anon.csv")
text = data['Anonymized Message']
encoder2 = OneHotEncoder()
encoder = LabelEncoder()
let_say =["how are you"]
now = encoder.fit_transform(let_say)
now2 = encoder2.fit_transform(now.reshape(-1,1))
#print now2

dex = encoder.fit_transform(text)
xex = encoder2.fit_transform(dex.reshape(-1,1))
print xex.toarray()
