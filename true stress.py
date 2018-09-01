import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import csv


data = pd.read_csv('filecom.csv')


eng_stress = data['eng_Stress']
eng_strain = data['eng_Strain']


plt.plot(eng_strain,eng_stress,'-g', label = 'Eng_stress,strain')
plt.xlabel('strain')
plt.ylabel('stress')


plt.show()
#lt.savefig('true and eng stress-strain graph')
