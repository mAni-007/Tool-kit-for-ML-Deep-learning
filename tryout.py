import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import csv


#data = pd.read_csv('MM362_group_2_COPPER.csv')


HV_two = ['658.6','390','347','333','327']
HV_two_depth = ['0','200','400','600','800']


HV_four = ['658.7','442.5','384.4','357.4','341.2']
HV_four_depth = ['0','200','400','600','800']



plt.plot(HV_four_depth,HV_four,'-g', label = '4_hour')
plt.xlabel('distance from circumference----->')
plt.ylabel('Hardness(HV)')
plt.plot(HV_two_depth,HV_two,'-b', label = '2_hour')
plt.legend(loc = 'upper right')

#plt.show()
plt.savefig('hardness graph')