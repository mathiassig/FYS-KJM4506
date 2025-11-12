import numpy as np
import matplotlib.pyplot as plt

def solid_angle(a,d):
    return 2*np.pi*(1-d/np.sqrt(d**2+a**2))

def epsilon_tot(N,A,t,I):
    return N/(A*t*I)

def epsilon_int(N,A,t,I,d,a):
    Omega = solid_angle(a,d)
    Total = epsilon_tot(N,A,t,I)
    return Total*4*np.pi/Omega

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

# area under gaussian
def events(amp,sigma):
    return amp*sigma*((2*np.pi)**0.5)


####### Distances, 137Cs #####################
distances = [5,10,15,20] # in cm
distance_errs = [0.1,0.1,0.1,0.1] # 5 mm error because of resolution of ruler
mus = []
mu_errs = []
sigmas = []
sigma_errs = []
amps = []
amp_errs = []
ts_dist = [289,295,297,298] # in seconds
tekst = [] # re-initilize tekst
tot_eff_dist = []
tot_eff_dist_errs = []
for i in range(len(distances)):
    A = activity(A0s[0],dts[0],T2s[0])*3.7*10**4 # calculate acitvity for 137Cs # convert from microCurie to disintegrations/sec
    N = events(amps[i],sigmas[i])
    tot_eff = epsilon_tot(N,A,ts_dist[i],0.8501)
    tot_eff_dist.append(tot_eff)
    tot_eff_dist_errs.append()
    Omega = solid_angle(2.54,distances[i])
    tekst.append(f"{Sources[0]} & {tot_eff} & {distances[i]} & {Omega}\\\\")

np.savetxt('tot_eff.txt', tekst, fmt='%s')

from scipy.optimize import curve_fit

def my_function(x, a,b,c):
    """Solid angle function"""
    return a *x/np.sqrt(x**2+b)+c
def my_function2(x,a):
    return a/(x**2)
xdata = np.arange(5,21,0.5)
popt, pcov = curve_fit(my_function, np.array(distances),np.array(tot_eff_dist))
plt.plot(xdata, my_function(xdata, *popt), color='red', label='Fitted Curve')
plt.errorbar(distances, tot_eff_dist, yerr=tot_eff_dist_errs,label= "137Cs 662keV peak", fmt='o',color='blue')
plt.title("Total peak efficiency as a function of \n distance between detector and source")
plt.xlabel('cm')
plt.ylabel("detector efficiency")
plt.legend()
plt.savefig("figures/tot_peak_eff_dist.png")
plt.close()


##### Energy peaks, all sources #################
Es = [81.0,356.0,662.0,1173.2,1332.5]
tot_eff_all = []
tot_eff_all_errs = []
Is = [0.3331,0.6205,0.8501,0.9988,0.9998]
ts_all = [293,293,289,298,298] # in seconds
mus = []
mu_errs = []
sigmas = []
sigma_errs = []
amps = []
amp_errs = []

source_indices = [1,1,0,2,2]
conversion_factors = [10**3,10**3,3.7*10**4,10**3,10**3]
tekst = [] # re-initialize tekst
for i in range(len(Es)):
    A = activity(A0s[source_indices[i]],dts[source_indices[i]],T2s[source_indices[i]])*conversion_factors[i]# calculate acitvity for sources, convert to disintegrations/sec
    N = events(amps[i],sigmas[i])
    tot_eff = epsilon_tot(N,A,ts_all[i],Is[i])
    tot_eff_all.append(tot_eff)
    int_eff = epsilon_int(N,A,ts_all[i],Is[i],distances[0],2.9)
    int_eff_err = 
    tekst.append(f"{Sources[source_indices[i]]} & {Es[i]} & {int_eff} \$ \pm \$ {int_eff_err}\\\\")
np.savetxt('int_eff.txt', tekst, fmt='%s')

plt.errorbar(Es, tot_eff_all, yerr=tot_eff_all_errs, fmt='o', color='blue')
plt.title("Total peak efficiency as a function of energy")
plt.xlabel("keV")
plt.ylabel("detector efficiency")
plt.savefig("figures/tot_peak_eff_all.png")
plt.close()