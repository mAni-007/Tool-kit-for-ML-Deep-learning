import quandl
import math
import numpy as np 

quandl.ApiConfig.api_key = 'ibDyF9TBEjBAAyQ6VgL8'
df = quandl.get('WIKI/GOOGL')


thefile = open('data.txt', 'w')
thefile.write(str(df))
thefile.close()