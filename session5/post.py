import numpy as np
import matplotlib.pyplot as plt

# area under gaussian
def events(amp):
    return amp # turns out we are using a function that is already normalized

def FWHM(sigma, sigma_err):
    fwhm = sigma*2*np.sqrt(2*np.log(2))
    return fwhm, fwhm*sigma_err/sigma # return value and error

def Res(sigma,sigma_err,E):
    fwhm,fwhm_err = FWHM(sigma,sigma_err)
    return fwhm/E*100,fwhm_err/E*100 # multiply by 100 to get resolution in percent

def Res_theoretical(E):
    epsilon = 2.95e-3# in keV, energy needed to create electron-hole pair
    F = 0.08 # Fano factor for HPGe
    return 2.35*np.sqrt(F*epsilon/E)*100 # multiply by 100 to get percentage

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

# switches
all_peaks = False
shaping_times = True
##### Energy peaks, all sources #################
if all_peaks:
    Es = [81.0,356.0,662.0,1173.2,1332.5]
    sigmas = [0.635, 0.639, 0.893,0.799, 0.843]
    sigma_errs = [0.030, 0.015, 0.038, 0.010, 0.012]
    resolutions = []
    resolution_errs = []

    tekst = [] # re-initialize tekst
    for i in range(len(Es)):
        resolution, resolution_err  =Res(sigmas[i],sigma_errs[i],Es[i])
        res_theory = Res_theoretical(Es[i])
        resolutions.append(resolution)
        resolution_errs.append(resolution_err)
        tekst.append(f" {Es[i]} & ${resolution:.3f}  \pm  {resolution_err:.3f}$ & {res_theory:.3f}\\\\")
    np.savetxt('resolution.txt', tekst, fmt='%s')

    from scipy.optimize import curve_fit

    def my_function(x,a):
        return a/x
    xdata = np.arange(70,1400,10)
    popt, pcov = curve_fit(my_function, np.array(Es),np.array(resolutions))
    plt.plot(xdata, my_function(xdata, *popt), color='red', label='Fitted curve')
    plt.errorbar(Es, resolutions, yerr=resolution_errs, fmt='.', color='blue', label='Energy peaks')
    plt.title("Resolution as a function of energy",fontsize=16)
    plt.xlabel("keV",fontsize=14)
    plt.ylabel("resolution [%]",fontsize=14)
    plt.legend()
    plt.savefig("figures/resolutions.png")
    plt.close()
if shaping_times:
    shapingtimes = [6,8,10]
    mus = [560.9,662.0,651.2]
    sigmas = [1.368,1.201,0.966]
    sigma_errs = [0.047,0.084,0.023]
    resolutions = []
    resolution_errs = []

    tekst = [] # re-initialize tekst
    for i in range(len(mus)):
        resolution, resolution_err  =Res(sigmas[i],sigma_errs[i],mus[i])
        res_theory = Res_theoretical(mus[i])
        resolutions.append(resolution)
        resolution_errs.append(resolution_err)
        tekst.append(f" {shapingtimes[i]} & ${resolution:.3f}  \pm  {resolution_err:.3f}$\\\\")
    np.savetxt('resolution_shapingtime.txt', tekst, fmt='%s')

    # from scipy.optimize import curve_fit

    # def my_function(x,a):
    #     return a/x
    # xdata = np.arange(70,1400,10)
    # popt, pcov = curve_fit(my_function, np.array(mus),np.array(resolutions))
    # plt.plot(xdata, my_function(xdata, *popt), color='red', label='Fitted curve')
    # plt.errorbar(mus, resolutions, yerr=resolution_errs, fmt='.', color='blue', label='Energy peaks')
    # plt.title("Resolution as a function of energy",fontsize=16)
    # plt.xlabel("keV",fontsize=14)
    # plt.ylabel("resolution [%]",fontsize=14)
    # plt.legend()
    # plt.savefig("figures/resolutions_shapingtime.png")
    # plt.close()