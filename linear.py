import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import csv


data = pd.read_csv('MM362_group_2_COPPER.csv')


eng_stress = data['log sigma']
eng_strain = data['log strain']
"""true_stress = data['True_stress']
true_strain = data['True_strain']"""

plt.plot(eng_strain,eng_stress,'-g', )
plt.xlabel('strain')
plt.ylabel('stress')
#plt.plot(true_strain,true_stress,'-b', )
plt.legend(loc = 'upper right')

plt.show()
#lt.savefig('true and eng stress-strain graph')