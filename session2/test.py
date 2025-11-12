import numpy as np
sourcename=input("Enter sourcename: (ex:239Pu)")
data = np.loadtxt(f'data/{sourcename}.txt',delimiter = ";",dtype=str)#load the data
print((data[1:,0]).astype(float))