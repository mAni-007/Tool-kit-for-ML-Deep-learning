from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing  import OneHotEncoder
import csv
import numpy as np 
import pandas as pd 


#We can apply both transformations (from text categories to integer categories, then from the integer categories to one-hot vectors)
data = pd.read_csv("dataset-fb-valence-arousal-anon.csv")
text = data['Anonymized Message']
encoder2 = OneHotEncoder()
encoder = LabelEncoder()
let_say =["the","how are you", "i'm a mani","how about you","just killing time","find me"]
now = encoder.fit_transform(let_say) # 1-D array
#print now
now2 = encoder2.fit_transform(now.reshape(-1,1)) #2-D array
#print now2.toarray(), now2



'''dex = encoder.fit_transform(text)
xex = encoder2.fit_transform(dex.reshape(-1,1))'''
#print xex.toarray()


#apply one-hot vector in one shot
 
from sklearn.preprocessing import  LabelBinarizer
encoder3 = LabelBinarizer()
one_shot = encoder3.fit_transform(let_say)
print one_shot