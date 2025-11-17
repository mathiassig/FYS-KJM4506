import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# peaks
## alpha lines ##
Pu_line1 = 5.499 # MeV
Pu_line2 = 5.457 # MeV
Cm_line1= 5.805 # MeV
Cm_line2= 5.763 # MeV

# switches
alpha_switch = True
beta_switch = False

# Definer en funksjon som beskriver kurven vi ønsker å tilpasse
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def line(x,a,b):
    return a*x+b

if alpha_switch:
    # kalibreringsdata
    x_data = np.array([434.439, 439.370, 480.446, 486.080])
    x_errors = np.array([0.604, 0.128, 0.231, 0.086])
    y_data = np.array([Pu_line2,Pu_line1,Cm_line2,Cm_line1])

    # Finne de beste parameterne for kurven
    params, covariance = curve_fit(line, x_data, y_data)

    # Ekstrahere parameterne
    a, b = params

    # Generer x-verdier for å plotte den tilpassede kurven
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = line(x_fit, a, b)

    # Plotting av data og tilpasset kurve
    plt.errorbar(x_data, y_data, xerr=x_errors, fmt='o',label='$\\alpha$ data', color='red')
    plt.plot(x_fit, y_fit, label='Fitted line', color='blue')
    plt.xlabel('Channel number',fontsize=14)
    plt.ylabel('MeV',fontsize=14)
    plt.title('Calibration linear fit',fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()
    print(f"a: {a}, b: {b}")

if beta_switch:
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