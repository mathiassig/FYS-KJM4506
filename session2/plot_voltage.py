import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(f'data/{"voltage.txt"}',delimiter = ";",dtype=str)#load the data

# plot the data
plt.plot((data[1:,0]).astype(float),(data[1:,1]).astype(float),label="N(V)")
plt.xlabel("Volt",fontsize=16)
plt.ylabel("Counts per 10s",fontsize=14)
plt.yscale("log")
plt.ylim(1.0e-1,2.0e+3)
plt.title("Number of counts as a function of voltage",fontsize=18)
plt.legend()
plt.savefig("figures/counts_per_voltage.png")
