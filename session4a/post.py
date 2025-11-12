import numpy as np
import matplotlib.pyplot as plt

# area under gaussian
def events(amp):
    return amp # turns out we are using a function that is already normalized

def events_err(amp_err):
    return amp_err # turns out we are using a function that is already normalized

def solid_angle(a,d):
    return 2*np.pi*(1-d/np.sqrt(d**2+a**2))

def solid_angle_err(a,d,d_err):
    return a**2/((d**2+a**2)**1.5)*d_err

def epsilon_tot(N,A,t,I):
    return N/(A*t*I)

def epsilon_tot_err(epsilon_tot,amp,amp_err,sigma,sigma_err):
    N_err = events_err(amp_err)
    N = events(amp)
    return epsilon_tot*N_err/N

def epsilon_int(N,A,t,I,d,a):
    Omega = solid_angle(a,d)
    Total = epsilon_tot(N,A,t,I)
    return Total*4*np.pi/Omega

def epsilon_int_err(epsilon_int,epsilon_tot,epsilon_tot_err,Omega,Omega_err):
    return epsilon_int*np.sqrt((epsilon_tot_err/epsilon_tot)**2+(Omega_err/Omega)**2)

def activity(A0,t,T2):
    lam = np.log(2)/T2
    return A0*np.exp(-lam*t)


Sources = ["137Cs","133Ba","60Co"]
Units = ["$mu$Ci","kBq","kBq"]
A0s = [11.46,160,16] # acitvity
T2s = [30.0,10.5,5.27] # half lives in years
dts = [47.0+8/12,23+9/12,23+9/12] # time elapsed in years
tekst = [] # tekst som skal skrives til fil
for i in range(len(A0s)):
    A = activity(A0s[i],dts[i],T2s[i])
    tekst.append(f"{Sources[i]} & {A} {Units[i]}\\\\")# format like latex table
# Skrive til fil
np.savetxt('activity.txt', tekst, fmt='%s')




####### Distances, 137Cs #####################
distances = [5,10,15,20] # in cm
distance_errs = [0.1,0.1,0.1,0.1] # 5 mm error because of resolution of ruler
mus = [672.6, 676.1, 678.2 , 678.8]
mu_errs = [0.2, 0.2, 0.2, 0.3]
sigmas = [28.3, 28.8 , 29.5, 29.4]
sigma_errs = [0.2, 0.2, 0.2, 0.3]
amps = [179382.63366313477, 79098.82120488651, 39831.3194202709, 24731.52684843751  ]
amp_errs = [923.8, 471.1, 257.5, 192.9]
ts_dist = [289,295,297,298] # in seconds
tekst = [] # re-initilize tekst
tot_eff_dist = []
tot_eff_dist_errs = []
for i in range(len(distances)):
    A = activity(A0s[0],dts[0],T2s[0])*3.7*10**4 # calculate acitvity for 137Cs # convert from microCurie to disintegrations/sec
    N = events(amps[i])
    tot_eff = epsilon_tot(N,A,ts_dist[i],0.8501)
    tot_eff_dist.append(tot_eff)
    tot_eff_dist_err = epsilon_tot_err(tot_eff,amps[i],amp_errs[i],sigmas[i],sigma_errs[i])
    tot_eff_dist_errs.append(tot_eff_dist_err)
    Omega = solid_angle(2.54,distances[i])
    Omega_err = solid_angle_err(2.54,distances[i],distance_errs[i])
    tekst.append(f"{Sources[0]} & {tot_eff} $\pm$ {tot_eff_dist_err} & {distances[i]} & {Omega} $\pm$ {Omega_err} \\\\")

np.savetxt('tot_eff.txt', tekst, fmt='%s')

from scipy.optimize import curve_fit

def my_function(x, a,b,c):
    """Solid angle function"""
    return a *x/np.sqrt(x**2+b)+c
def my_function2(x,a):
    "simple parabola"
    return a/(x**2)
def my_function3(x,a,b):
    "line"
    return a*x+b
xdata = np.arange(5,21,0.5)
popt, pcov = curve_fit(my_function2, np.array(distances),np.array(tot_eff_dist))
plt.plot(xdata, my_function2(xdata, *popt), color='red', label='Fitted Curve')
plt.errorbar(distances, tot_eff_dist, yerr=tot_eff_dist_errs,label= "137Cs 662keV peak", fmt='o',color='blue')
plt.title("Total peak efficiency as a function of \n distance between detector and source",fontsize=16)
plt.xlabel('cm',fontsize=14)
plt.ylabel("detector efficiency",fontsize=14)
plt.legend()
plt.savefig("figures/tot_peak_eff_dist.png")
plt.close()


##### Energy peaks, all sources #################
Es = [81.0,356.0,662.0,1173.2,1332.5]
tot_eff_all = []
tot_eff_all_errs = []
Is = [0.3331,0.6205,0.8501,0.9988,0.9998]
ts_all = [293,293,289,298,298] # in seconds
mus = [62.7, 358.0, 672.6, 1202.7, 1360.3]
mu_errs = [0.1, 0.5, 0.2, 1.5, 1.7]
sigmas = [7.1, 24.6, 28.3, 46.5, 48.7]
sigma_errs = [0.1, 0.5, 0.2, 1.7, 2.1]
amps = [76735.79211938973, 84512.9773697688, 79098.82120488651, 2517.7377997181443, 2181.6313046715945]
amp_errs = [816.6, 1413.7, 471.1, 73.7, 73.0]

source_indices = [1,1,0,2,2]
conversion_factors = [10**3,10**3,3.7*10**4,10**3,10**3]
tekst = [] # re-initialize tekst
for i in range(len(Es)):
    A = activity(A0s[source_indices[i]],dts[source_indices[i]],T2s[source_indices[i]])*conversion_factors[i]# calculate acitvity for sources, convert to disintegrations/sec
    N = events(amps[i])
    tot_eff = epsilon_tot(N,A,ts_all[i],Is[i])
    tot_eff_all.append(tot_eff)
    tot_eff_err = epsilon_tot_err(tot_eff,amps[i],amp_errs[i],sigmas[i],sigma_errs[i])
    tot_eff_all_errs.append(tot_eff_err)
    int_eff = epsilon_int(N,A,ts_all[i],Is[i],distances[0],2.9)
    Omega = solid_angle(2.54,5)
    Omega_err = solid_angle_err(2.54,5,0.1)
    int_eff_err = epsilon_int_err(int_eff,tot_eff,tot_eff_err,Omega,Omega_err)
    tekst.append(f"{Sources[source_indices[i]]} & {Es[i]} & {int_eff} \$ \pm \$ {int_eff_err}\\\\")
np.savetxt('int_eff.txt', tekst, fmt='%s')

xdata = np.arange(0,1400,100)
popt, pcov = curve_fit(my_function3, np.array(Es),np.array(tot_eff_all))
plt.plot(xdata, my_function3(xdata, *popt), color='red', label='Fitted line')
plt.errorbar(Es, tot_eff_all, yerr=tot_eff_all_errs, fmt='o', color='blue', label='energy peaks')
plt.title("Total peak efficiency as a function of energy",fontsize=16)
plt.xlabel("keV",fontsize=14)
plt.ylabel("detector efficiency",fontsize=14)
plt.legend()
plt.savefig("figures/tot_peak_eff_all.png")
plt.close()