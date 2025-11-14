import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# peaks
Ba_gamma_1 = 81.0 # keV
Ba_gamma_2 = 356 # keV
Cs_gamma = 661.659  # keV.
Co_gamma_1 = 1173.2   # keV.
Co_gamma_2 = 1332.5   # keV.
Eu_gamma_1 = 121.78300 # +-0.002 # keV
Eu_gamma_2 = 1408.0110 # +- 0.0140  # keV
Eu_gamma_1_err = 0.002
Eu_gamma_2_err = 0.0140 

# switches
HPGe_switch = False
NaI_switch = True

# Definer en funksjon som beskriver kurven vi ønsker å tilpasse
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def line(x,a,b):
    return a*x+b

if HPGe_switch:
    # kalibreringsdata
    x_data = np.array([3482.4, 6173.7, 7011.7])
    x_errors = np.array([0.4, 0.6, 0.6])
    y_data = np.array([Cs_gamma,Co_gamma_1,Co_gamma_2])

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

if NaI_switch:
    # kalibreringsdata
    x_data = np.array([25.3, 425.4])
    x_errors = np.array([0.1, 0.0])
    y_data = np.array([Eu_gamma_1,Eu_gamma_2])
    y_errors = np.array([Eu_gamma_1_err,Eu_gamma_2_err])

    # Finne de beste parameterne for kurven
    params, covariance = curve_fit(line, x_data, y_data)

    # Ekstrahere parameterne
    a, b = params

    # Generer x-verdier for å plotte den tilpassede kurven
    x_fit = np.linspace(min(x_data), max(x_data), 10)
    y_fit = line(x_fit, a, b)

    # Plotting av data og tilpasset kurve
    plt.errorbar(x_data, y_data, xerr=x_errors,yerr=y_errors, fmt='o',label='Data', color='red')
    plt.plot(x_fit, y_fit, label='Fitted line', color='blue')
    plt.xlabel('channel number',fontsize=14)
    plt.ylabel('keV',fontsize=14)
    plt.title('Calibration linear fit NaI(Tl)',fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()
    print(f"a: {a}, b: {b}")