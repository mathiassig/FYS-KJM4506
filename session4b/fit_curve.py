import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# peaks
Ba_gamma_1 = 81.0 # keV
Ba_gamma_2 = 356 # keV
Cs_gamma = 661.659  # keV.
Co_gamma_1 = 1173.2   # keV.
Co_gamma_2 = 1332.5   # keV.

# kalibreringsdata
x_data = np.array([3482.4, 6173.7, 7011.7])
x_errors = np.array([0.4, 0.6, 0.6])
y_data = np.array([Cs_gamma,Co_gamma_1,Co_gamma_2])

# Definer en funksjon som beskriver kurven vi ønsker å tilpasse
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def line(x,a,b):
    return a*x+b

# Finne de beste parameterne for kurven
params, covariance = curve_fit(line, x_data, y_data)

# Ekstrahere parameterne
a, b = params

# Generer x-verdier for å plotte den tilpassede kurven
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = line(x_fit, a, b)

# Plotting av data og tilpasset kurve
plt.errorbar(x_data, y_data, xerr=x_errors, fmt='o',label='Data', color='red')
plt.plot(x_fit, y_fit, label='Fitted line', color='blue')
plt.xlabel('channel number',fontsize=14)
plt.ylabel('keV',fontsize=14)
plt.title('Calibration linear fit',fontsize=16)
plt.legend()
plt.grid()
plt.show()
print(f"a: {a}, b: {b}")