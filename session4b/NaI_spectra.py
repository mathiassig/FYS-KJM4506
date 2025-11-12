import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.stats as ss

def read_spectrum(file_path: str) -> np.ndarray:
    
    result: list = []
    with open(file_path) as f:
        npoints: int = 0
        i: int = 0
        at_data: bool = False

        for l in f:
            if not at_data and "$DATA" in l:
                at_data = True
                continue
            elif not at_data:
                continue

            if at_data and npoints == 0:
                ls: list[str] = l.split()
                npoints = int(ls[1])
                continue

            result.append(float(l))
            i += 1

            if i > npoints:
                break

    return np.asarray(result)

def read_measurement_time(file_path: str) -> int:
    with open(file_path) as f:
        at_meas_tim: bool = False

        for l in f:
            if not at_meas_tim and "$MEAS_TIM" in l:
                at_meas_tim = True
                continue
            elif at_meas_tim:
                return int(l.split()[0])
            
    return 0

def pol1(x: npt.ArrayLike, a: float, b: float):
    x_array: np.ndarray = np.asarray(x)
    return a * x_array + b
 
def pol1_err(x: npt.ArrayLike, a_err: float, b_err: float):
    x_array: np.ndarray = np.asarray(x)
    return np.sqrt( np.power(x_array * a_err, 2) + b_err * b_err)

def inv_pol1(x: npt.ArrayLike, a: float, b: float):
    x_array: np.ndarray = np.asarray(x)
    return (x_array - b) / a

def exp_gaus(x: npt.ArrayLike, A: float, lam: float, k: float, mu: float, sigma: float) -> np.ndarray:
    x_array: np.ndarray = np.asarray(x, dtype=np.float64)

    return A * np.exp(-lam * x_array) + k * np.exp( -0.5 * np.power( (x_array - mu) / sigma, 2) )

def exp_2gaus(x: npt.ArrayLike, A: float, lam: float, k1: float, mu1: float, sigma1: float, k2: float, mu2: float, sigma2: float) -> np.ndarray:
    x_array: np.ndarray = np.asarray(x, dtype=np.float64)

    return A * np.exp(-lam * x_array) + k1 * np.exp( -0.5 * np.power( (x_array - mu1) / sigma1, 2) ) + k2 * np.exp( -0.5 * np.power( (x_array - mu2) / sigma2, 2) )

def exp_3gaus(x: npt.ArrayLike, A: float, lam: float, k1: float, mu1: float, sigma1: float, k2: float, mu2: float, sigma2: float, k3: float, mu3: float, sigma3: float, ) -> np.ndarray:
    x_array: np.ndarray = np.asarray(x, dtype=np.float64)

    return A * np.exp(-lam * x_array) + k1 * np.exp( -0.5 * np.power( (x_array - mu1) / sigma1, 2) ) \
        + k2 * np.exp( -0.5 * np.power( (x_array - mu2) / sigma2, 2) ) + k3 * np.exp( -0.5 * np.power( (x_array - mu3) / sigma3, 2) )

def main():
    plt.rcParams.update({'text.usetex': False})
    plt.rcParams.update({'ytick.direction': 'in'})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.right': True})
    plt.rcParams.update({'xtick.top': True})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 16})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'legend.fontsize': 16})

    # Background spectrum analysis

    background_spec: np.ndarray = read_spectrum("spectra/background.Spe")
    background_spec_err: np.ndarray = np.sqrt(background_spec) # when dealing with counts, the statistical error is always the square root of the number of counts
    background_meas_time: int = read_measurement_time("spectra/background.Spe")
    meas_time_error: float = 0.5 # all measturement times are given in integer seconds, thus we have an error of 0.5 s

    nbins: int = background_spec.size # this is the same for all spectra
    bin_low_edges: np.ndarray = np.linspace(0, nbins - 1, nbins) # this is also the same for all spectra

    background_fit_range: tuple[int, int] = (1400, 1800)
    background_fit_nbins: int = background_fit_range[1] - background_fit_range[0]
    background_fit_bin_centers: np.ndarray = np.linspace(background_fit_range[0], background_fit_range[1] - 1, background_fit_nbins) + 0.5
    background_fit_spec: np.ndarray = background_spec[background_fit_range[0]: background_fit_range[1]]
    background_fit_spec_err: np.ndarray = background_spec_err[background_fit_range[0]: background_fit_range[1]]

    background_fit_spec_nonzero = background_fit_spec.nonzero()[0] # get indices of bins with nonzero counts

    # we are ignoring any bins with zero counts
    background_fit_bin_centers: np.ndarray = background_fit_bin_centers[background_fit_spec_nonzero]
    background_fit_spec: np.ndarray = background_fit_spec[background_fit_spec_nonzero]
    background_fit_spec_err: np.ndarray = background_fit_spec_err[background_fit_spec_nonzero]
    background_fit_nbins: int = background_fit_spec_nonzero.size

    background_fit_guess: list[float] = [5, 0.0005, 10, 1630, 20]

    background_p_opt: np.ndarray
    background_p_cov: np.ndarray
    try:
        background_p_opt, background_p_cov = so.curve_fit(exp_gaus, background_fit_bin_centers, background_fit_spec, background_fit_guess, background_fit_spec_err, absolute_sigma=True) # absolute sigma should always be set to true
        background_p_err: np.ndarray = np.sqrt(np.diag(background_p_cov))
    except: # if the fit fails the the scrip will continue using the guess as the fit parameters with 0 error
        print("Background fit failed")
        background_p_opt = np.asarray(background_fit_guess)
        background_p_err = np.asarray([0, 0, 0, 0, 0])

    background_fit: np.ndarray = exp_gaus(background_fit_bin_centers, *background_p_opt)

    background_fit_chi2: float = np.sum( np.power( (background_fit_spec - background_fit), 2) / background_fit )
    background_fit_dof: int = background_fit_nbins - 5
    background_fit_P: float = float(ss.chi2.sf(background_fit_chi2, background_fit_dof))

    print("Background fit output:")
    print(f"  chi2: {background_fit_chi2 : .2f}")
    print(f"  chi2 / dof: {background_fit_chi2 / background_fit_dof : .2f}")
    print(f"  P value: {background_fit_P : 2e}")
    print(f"Fit parameters:")
    print(f"  A: {background_p_opt[0] : .2f} +/- {background_p_err[0] : .2f}")
    print(f"  lambda: {background_p_opt[1] : .5f} +/- {background_p_err[1] : .5f}")
    print(f"  k: {background_p_opt[2] : .2f} +/- {background_p_err[2] : .2f}")
    print(f"  mu: {background_p_opt[3] : .2f} +/- {background_p_err[3] : .2f}")
    print(f"  sigma: {background_p_opt[4] : .2f} +/- {background_p_err[4] : .2f}")

    # for plotting the fit with more points
    background_fit_x: np.ndarray = np.linspace(background_fit_range[0], background_fit_range[1], 1000)
    background_fit: np.ndarray = exp_gaus(background_fit_x, *background_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges, nbins, range=(0, nbins), weights=background_spec, histtype="step", label="Background")
    ax.plot(background_fit_x, background_fit, color="black", label="Fit")
    ax.set_ylabel("Counts / channel")
    ax.set_xlabel("Channel number")
    ax.legend(loc="upper right")

    fig.savefig("NaI_background.png", dpi=400, bbox_inches="tight")

    # Cesium spectrum analysis

    cesium_spec: np.ndarray = read_spectrum("spectra/cesium.Spe")
    cesium_spec_err: np.ndarray = np.sqrt(cesium_spec) # when dealing with counts, the statistical error is always the square root of the number of counts
    cesium_meas_time: int = read_measurement_time("spectra/cesium.Spe")

    cesium_background_ratio: float = cesium_meas_time / background_meas_time
    cesium_background_ratio_err: float = meas_time_error * np.sqrt( np.power(1 / background_meas_time, 2) + np.power(cesium_meas_time / background_meas_time**2, 2) )

    cesium_spec_bs: np.ndarray = cesium_spec -  background_spec * cesium_background_ratio
    for i in range(nbins):
        if cesium_spec_bs[i] < 0:
            cesium_spec_bs[i] = 0
    
    cesium_spec_bs_err: np.ndarray = np.sqrt( np.power(cesium_spec_err, 2) + np.power(background_spec_err * cesium_background_ratio, 2) 
                                             + np.power(background_spec * cesium_background_ratio_err, 2) ) # error propagation for background subtracted spectrum

    cesium_bs_fit_range: tuple[int, int] = (650, 900)
    cesium_bs_fit_nbins: int = cesium_bs_fit_range[1] - cesium_bs_fit_range[0]
    cesium_bs_fit_bin_centers: np.ndarray = np.linspace(cesium_bs_fit_range[0], cesium_bs_fit_range[1] - 1, cesium_bs_fit_nbins) + 0.5
    cesium_bs_fit_spec: np.ndarray = cesium_spec[cesium_bs_fit_range[0]: cesium_bs_fit_range[1]]
    cesium_bs_fit_spec_err: np.ndarray = cesium_spec_bs_err[cesium_bs_fit_range[0]: cesium_bs_fit_range[1]]

    cesium_bs_fit_guess: list[float] = [7e3, 0.005, 1.1e4, 770, 20]

    cesium_p_opt: np.ndarray
    cesium_p_cov: np.ndarray
    try:
        cesium_p_opt, cesium_p_cov = so.curve_fit(exp_gaus, cesium_bs_fit_bin_centers, cesium_bs_fit_spec, cesium_bs_fit_guess, cesium_bs_fit_spec_err, absolute_sigma=True) # absolute sigma should always be set to true
        cesium_p_err: np.ndarray = np.sqrt(np.diag(cesium_p_cov))
    except: # if the fit fails the the scrip will continue using the guess as the fit parameters with 0 error
        print("Cesium fit failed")
        cesium_p_opt = np.asarray(cesium_bs_fit_guess)
        cesium_p_err = np.asarray([0, 0, 0, 0, 0])

    cesium_bs_fit: np.ndarray = exp_gaus(cesium_bs_fit_bin_centers, *cesium_p_opt)

    cesium_bs_fit_chi2: float = np.sum( np.power( (cesium_bs_fit_spec - cesium_bs_fit), 2) / cesium_bs_fit )
    cesium_bs_fit_dof: int = cesium_bs_fit_nbins - 5
    cesium_bs_fit_P: float = float(ss.chi2.sf(cesium_bs_fit_chi2, cesium_bs_fit_dof))

    print("Cesium fit output:")
    print(f"  chi2: {cesium_bs_fit_chi2 : .2f}")
    print(f"  chi2 / dof: {cesium_bs_fit_chi2 / cesium_bs_fit_dof : .2f}")
    print(f"  P value: {cesium_bs_fit_P : 2e}")
    print(f"Fit parameters:")
    print(f"  A: {cesium_p_opt[0] : .2f} +/- {cesium_p_err[0] : .2f}")
    print(f"  lambda: {cesium_p_opt[1] : .5f} +/- {cesium_p_err[1] : .5f}")
    print(f"  k: {cesium_p_opt[2] : .2f} +/- {cesium_p_err[2] : .2f}")
    print(f"  mu: {cesium_p_opt[3] : .2f} +/- {cesium_p_err[3] : .2f}")
    print(f"  sigma: {cesium_p_opt[4] : .2f} +/- {cesium_p_err[4] : .2f}")


    cesium_bs_fit_x: np.ndarray = np.linspace(cesium_bs_fit_range[0], cesium_bs_fit_range[1], 1000)
    cesium_bs_fit: np.ndarray = exp_gaus(cesium_bs_fit_x, *cesium_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges, nbins, range=(0, nbins), weights=cesium_spec_bs, histtype="step", label="$^{137}$Cs - background")
    ax.hist(bin_low_edges, nbins, range=(0, nbins), weights=cesium_spec, histtype="step", label="$^{137}$Cs")
    ax.plot(cesium_bs_fit_x, cesium_bs_fit, color="black", label="Fit")
    ax.set_ylabel("Counts / channel")
    ax.set_xlabel("Channel number")
    ax.legend(loc="upper right")

    fig.savefig("NaI_cesium.png", dpi=400, bbox_inches="tight")

    # Cobolt spectrum analysis

    cobolt_spec: np.ndarray = read_spectrum("spectra/cobolt.Spe")
    cobolt_spec_err: np.ndarray = np.sqrt(cobolt_spec) # when dealing with counts, the statistical error is always the square root of the number of counts
    cobolt_meas_time: int = read_measurement_time("spectra/cobolt.Spe")

    cobolt_background_ratio: float = cobolt_meas_time / background_meas_time
    cobolt_background_ratio_err: float = meas_time_error * np.sqrt( np.power(1 / background_meas_time, 2) + np.power(cobolt_meas_time / background_meas_time**2, 2) )

    cobolt_spec_bs: np.ndarray = cobolt_spec -  background_spec * cobolt_background_ratio
    for i in range(nbins):
        if cobolt_spec_bs[i] < 0:
            cobolt_spec_bs[i] = 0
    
    cobolt_spec_bs_err: np.ndarray = np.sqrt( np.power(cobolt_spec_err, 2) + np.power(background_spec_err * cobolt_background_ratio, 2) 
                                             + np.power(background_spec * cobolt_background_ratio_err, 2) ) # error propagation for background subtracted spectrum

    cobolt_bs_fit_range: tuple[int, int] = (1100, 1600)
    cobolt_bs_fit_nbins: int = cobolt_bs_fit_range[1] - cobolt_bs_fit_range[0]
    cobolt_bs_fit_bin_centers: np.ndarray = np.linspace(cobolt_bs_fit_range[0], cobolt_bs_fit_range[1] - 1, cobolt_bs_fit_nbins) + 0.5
    cobolt_bs_fit_spec: np.ndarray = cobolt_spec[cobolt_bs_fit_range[0]: cobolt_bs_fit_range[1]]
    cobolt_bs_fit_spec_err: np.ndarray = cobolt_spec_bs_err[cobolt_bs_fit_range[0]: cobolt_bs_fit_range[1]]

    cobolt_bs_fit_spec_nonzero = cobolt_bs_fit_spec.nonzero()[0] # get indices of bins with nonzero counts

    # we are ignoring any bins with zero counts
    cobolt_bs_fit_bin_centers: np.ndarray = cobolt_bs_fit_bin_centers[cobolt_bs_fit_spec_nonzero]
    cobolt_bs_fit_spec: np.ndarray = cobolt_bs_fit_spec[cobolt_bs_fit_spec_nonzero]
    cobolt_bs_fit_spec_err: np.ndarray = cobolt_bs_fit_spec_err[cobolt_bs_fit_spec_nonzero]
    cobolt_bs_fit_nbins: int = cobolt_bs_fit_spec_nonzero.size

    cobolt_bs_fit_guess: list[float] = [7e3, 0.005, 20, 1300, 20, 20, 1500, 20]

    cobolt_p_opt: np.ndarray
    cobolt_p_cov: np.ndarray
    try:
        cobolt_p_opt, cobolt_p_cov = so.curve_fit(exp_2gaus, cobolt_bs_fit_bin_centers, cobolt_bs_fit_spec, cobolt_bs_fit_guess, cobolt_bs_fit_spec_err, absolute_sigma=True) # absolute sigma should always be set to true
        cobolt_p_err: np.ndarray = np.sqrt(np.diag(cobolt_p_cov))
    except: # if the fit fails the the scrip will continue using the guess as the fit parameters with 0 error
        print("Cobolt fit failed")
        cobolt_p_opt = np.asarray(cobolt_bs_fit_guess)
        cobolt_p_err = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

    cobolt_bs_fit: np.ndarray = exp_2gaus(cobolt_bs_fit_bin_centers, *cobolt_p_opt)

    cobolt_bs_fit_chi2: float = np.sum( np.power( (cobolt_bs_fit_spec - cobolt_bs_fit), 2) / cobolt_bs_fit )
    cobolt_bs_fit_dof: int = cobolt_bs_fit_nbins - 5
    cobolt_bs_fit_P: float = float(ss.chi2.sf(cobolt_bs_fit_chi2, cobolt_bs_fit_dof))

    print("Cobolt fit output:")
    print(f"  chi2: {cobolt_bs_fit_chi2 : .2f}")
    print(f"  chi2 / dof: {cobolt_bs_fit_chi2 / cobolt_bs_fit_dof : .2f}")
    print(f"  P value: {cobolt_bs_fit_P : 2e}")
    print(f"Fit parameters:")
    print(f"  A: {cobolt_p_opt[0] : .2f} +/- {cobolt_p_err[0] : .2f}")
    print(f"  lambda: {cobolt_p_opt[1] : .5f} +/- {cobolt_p_err[1] : .5f}")
    print(f"  k1: {cobolt_p_opt[2] : .2f} +/- {cobolt_p_err[2] : .2f}")
    print(f"  mu1: {cobolt_p_opt[3] : .2f} +/- {cobolt_p_err[3] : .2f}")
    print(f"  sigma1: {cobolt_p_opt[4] : .2f} +/- {cobolt_p_err[4] : .2f}")
    print(f"  k2: {cobolt_p_opt[5] : .2f} +/- {cobolt_p_err[5] : .2f}")
    print(f"  mu2: {cobolt_p_opt[6] : .2f} +/- {cobolt_p_err[6] : .2f}")
    print(f"  sigma2: {cobolt_p_opt[7] : .2f} +/- {cobolt_p_err[7] : .2f}")

    cobolt_bs_fit_x: np.ndarray = np.linspace(cobolt_bs_fit_range[0], cobolt_bs_fit_range[1], 1000)
    cobolt_bs_fit: np.ndarray = exp_2gaus(cobolt_bs_fit_x, *cobolt_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges, nbins, range=(0, nbins), weights=cobolt_spec_bs, histtype="step", label="$^{60}$Co - background")
    ax.hist(bin_low_edges, nbins, range=(0, nbins), weights=cobolt_spec, histtype="step", label="$^{60}$Co")
    ax.plot(cobolt_bs_fit_x, cobolt_bs_fit, color="black", label="Fit")
    ax.set_ylabel("Counts / channel")
    ax.set_xlabel("Channel number")
    ax.legend(loc="upper right")

    fig.savefig("NaI_cobolt.png", dpi=400, bbox_inches="tight")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={"hspace": 0})
    axs[0].hist(bin_low_edges, nbins, range=(0, nbins), weights=background_spec, histtype="step", label="Background")
    axs[1].hist(bin_low_edges, nbins, range=(0, nbins), weights=cesium_spec, histtype="step", label="$^{137}$Cs")
    axs[2].hist(bin_low_edges, nbins, range=(0, nbins), weights=cobolt_spec, histtype="step", label="$^{60}$Co")
    axs[0].set_ylabel("Counts")
    axs[1].set_ylabel("Counts")
    axs[2].set_ylabel("Counts")
    axs[2].set_xlabel("Channel number")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper left")

    fig.savefig("NaI_spectra_raw.png", dpi=400, bbox_inches="tight")

    # Calibrate

    calibration_x: np.ndarray = np.asarray([cesium_p_opt[3], cobolt_p_opt[3], cobolt_p_opt[6], background_p_opt[3]], dtype=np.float64)
    calibration_x_err: np.ndarray = np.asarray([cesium_p_err[3], cobolt_p_err[3], cobolt_p_err[6], background_p_err[3]], dtype=np.float64)

    calibration_y: np.ndarray = np.asarray([661.659, 1173.228, 1332.514, 1460.820], dtype=np.float64) # error is small enough to be neglected

    calibration_p_opt: np.ndarray
    calibration_p_cov: np.ndarray

    calibration_p_opt, calibration_p_cov = so.curve_fit(inv_pol1, calibration_y, calibration_x, sigma=calibration_x_err, absolute_sigma=True)
    calibration_p_err: np.ndarray = np.sqrt(np.diag(calibration_p_cov))

    calibration_fit: np.ndarray = inv_pol1(calibration_y, *calibration_p_opt)

    calibration_fit_chi2: float = np.sum( np.power((calibration_x - calibration_fit) / calibration_x_err, 2) )
    calibration_fit_dof: int = calibration_x.size - 2
    calibration_fit_P: float = float(ss.chi2.sf(calibration_fit_chi2, calibration_fit_dof))


    print("Calibration fit output:")
    print(f"  chi2: {calibration_fit_chi2 : .2f}")
    print(f"  chi2 / dof: {calibration_fit_dof : .2f}")
    print(f"  P value: {calibration_fit_P : 2e}")
    print("Fit parameters:")
    print(f"  a: {calibration_p_opt[0] : .4f} +/- {calibration_p_err[0] : .4f}")
    print(f"  b: {calibration_p_opt[1] : .2f} +/- {calibration_p_err[1] : .2f}")

    calibration_fit_x: np.ndarray = np.linspace(0, nbins, 1000)
    calibration_fit = pol1(calibration_fit_x, *calibration_p_opt)
    calibration_fit_err = pol1_err(calibration_fit_x, *calibration_p_err)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.errorbar(calibration_x, calibration_y, xerr=calibration_x_err, linestyle="none", marker=".", capsize=5, color="red", label="Data")
    ax.plot(calibration_fit_x, calibration_fit, color="black", label="Fit")
    ax.fill_between(calibration_fit_x, calibration_fit + calibration_fit_err, calibration_fit - calibration_fit_err, color="orange", alpha=0.5, label="Fit error")
    ax.set_ylabel("Energy [keV]")
    ax.set_xlabel("Channel number")
    ax.legend(loc="upper left")

    fig.savefig("NaI_calibration.png", dpi=400, bbox_inches="tight")


    bin_width: float = calibration_p_opt[0]
    xmin: float = calibration_p_opt[1] + (abs(calibration_p_opt[1] / bin_width) + 1) * bin_width
    xmax: float = float(pol1(nbins, *calibration_p_opt))
    nbins = round( (xmax - xmin) / bin_width )

    bin_low_edges_cal: np.ndarray = pol1(bin_low_edges, *calibration_p_opt)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={"hspace": 0})
    axs[0].hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=background_spec, histtype="step", label="Background")
    axs[1].hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=cesium_spec_bs, histtype="step", label="$^{137}$Cs - background")
    axs[2].hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=cobolt_spec_bs, histtype="step", label="$^{60}$Co - background")
    axs[0].set_ylabel("Counts")
    axs[1].set_ylabel("Counts")
    axs[2].set_ylabel("Counts")
    axs[2].set_xlabel("Energy [keV]")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper left")

    fig.savefig("NaI_spectra_calibrated.png", dpi=400, bbox_inches="tight")

    #plt.show()
    
    # Cesium peak resolution

    cesium_cal_fit_range: tuple[int, int] = (int((500 - bin_low_edges_cal[0]) / bin_width), int((800 - bin_low_edges_cal[0]) / bin_width))
    cesium_cal_fit_bin_centers: np.ndarray = bin_low_edges_cal[cesium_cal_fit_range[0] : cesium_cal_fit_range[1]] + bin_width / 2

    cesium_cal_fit_spec_bs: np.ndarray = cesium_spec_bs[cesium_cal_fit_range[0] : cesium_cal_fit_range[1]]
    cesium_cal_fit_spec_bs_err: np.ndarray = cesium_spec_bs_err[cesium_cal_fit_range[0]: cesium_cal_fit_range[1]]

    cesium_cal_bs_fit_guess: list[float] = [7e3, 0.005, 1.1e4, 662, 20]

    try:
        cesium_cal_p_opt, cesium_cal_p_cov = so.curve_fit(exp_gaus, cesium_cal_fit_bin_centers, cesium_cal_fit_spec_bs, cesium_cal_bs_fit_guess, 
                                                          cesium_cal_fit_spec_bs_err, absolute_sigma=True)
        cesium_cal_p_err: np.ndarray = np.sqrt(np.diag(cesium_cal_p_cov))
    except:
        print("Cesium calibrated fit failed")
        cesium_cal_p_opt = np.asarray(cesium_cal_bs_fit_guess)
        cesium_cal_p_err = np.asarray([0, 0, 0, 0, 0])

    cesium_resolution: float = 100 * np.sqrt(8 * np.log(2)) * cesium_cal_p_opt[4] / cesium_cal_p_opt[3]
    cesium_resolution_err: float = 100 * np.sqrt(8 * np.log(2)) * np.sqrt( np.power(cesium_cal_p_err[4] / cesium_cal_p_opt[3], 2)
                                                 + np.power(cesium_cal_p_opt[4] / cesium_cal_p_opt[3]**2 * cesium_cal_p_err[3], 2))

    print(f"137Cs peak resolution (662 keV):  {cesium_resolution : .3f} +/- {cesium_resolution_err : .3f}%")

    cesium_cal_bs_fit_x: np.ndarray = np.linspace(cesium_cal_fit_bin_centers[0], cesium_cal_fit_bin_centers[-1], 1000)
    cesium_cal_bs_fit: np.ndarray = exp_gaus(cesium_cal_bs_fit_x, *cesium_cal_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=cesium_spec_bs, histtype="step", label="$^{137}$Cs - background")
    ax.plot(cesium_cal_bs_fit_x, cesium_cal_bs_fit, color="black", label="Fit")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Energy [keV]")
    ax.legend(loc="upper right")

    fig.savefig("NaI_cesium_cal.png", dpi=400, bbox_inches="tight")

    # Cobolt peaks resolution

    cobolt_cal_fit_range: tuple[int, int] = (int((1000 - bin_low_edges_cal[0]) / bin_width), int((1500 - bin_low_edges_cal[0]) / bin_width))
    cobolt_cal_fit_bin_centers: np.ndarray = bin_low_edges_cal[cobolt_cal_fit_range[0] : cobolt_cal_fit_range[1]] + bin_width / 2

    cobolt_cal_fit_spec_bs: np.ndarray = cobolt_spec_bs[cobolt_cal_fit_range[0] : cobolt_cal_fit_range[1]]
    cobolt_cal_fit_spec_bs_err: np.ndarray = cobolt_spec_bs_err[cobolt_cal_fit_range[0]: cobolt_cal_fit_range[1]]

    cobolt_cal_fit_spec_bs_nonzero: np.ndarray = cobolt_cal_fit_spec_bs.nonzero()[0]

    cobolt_cal_fit_spec_bs = cobolt_cal_fit_spec_bs[cobolt_cal_fit_spec_bs_nonzero]
    cobolt_cal_fit_spec_bs_err = cobolt_cal_fit_spec_bs_err[cobolt_cal_fit_spec_bs_nonzero]
    cobolt_cal_fit_bin_centers = cobolt_cal_fit_bin_centers[cobolt_cal_fit_spec_bs_nonzero]

    cobolt_cal_bs_fit_guess: list[float] = [7e3, 0.005, 20, 1170, 20, 20, 1333, 20]

    try:
        cobolt_cal_p_opt, cobolt_cal_p_cov = so.curve_fit(exp_2gaus, cobolt_cal_fit_bin_centers, cobolt_cal_fit_spec_bs, cobolt_cal_bs_fit_guess, 
                                                          cobolt_cal_fit_spec_bs_err, absolute_sigma=True)
        cobolt_cal_p_err: np.ndarray = np.sqrt(np.diag(cobolt_cal_p_cov))
    except:
        print("cobolt calibrated fit failed")
        cobolt_cal_p_opt = np.asarray(cobolt_cal_bs_fit_guess)
        cobolt_cal_p_err = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

    cobolt1_resolution: float = 100 * np.sqrt(8 * np.log(2)) * cobolt_cal_p_opt[4] / cobolt_cal_p_opt[3]
    cobolt1_resolution_err: float = 100 * np.sqrt(8 * np.log(2)) * np.sqrt( np.power(cobolt_cal_p_err[4] / cobolt_cal_p_opt[3], 2)
                                                 + np.power(cobolt_cal_p_opt[4] / cobolt_cal_p_opt[3]**2 * cobolt_cal_p_err[3], 2))
    
    cobolt2_resolution: float = 100 * np.sqrt(8 * np.log(2)) * cobolt_cal_p_opt[7] / cobolt_cal_p_opt[6]
    cobolt2_resolution_err: float = 100 * np.sqrt(8 * np.log(2)) * np.sqrt( np.power(cobolt_cal_p_err[7] / cobolt_cal_p_opt[6], 2)
                                                 + np.power(cobolt_cal_p_opt[7] / cobolt_cal_p_opt[6]**2 * cobolt_cal_p_err[6], 2))

    print(f"60Co peak 1 resolution (1173 keV):  {cobolt1_resolution : .3f} +/- {cobolt1_resolution_err : .3f}%")
    print(f"60Co peak 2 resolution (1333 keV):  {cobolt2_resolution : .3f} +/- {cobolt2_resolution_err : .3f}%")

    cobolt_cal_bs_fit_x: np.ndarray = np.linspace(cobolt_cal_fit_bin_centers[0], cobolt_cal_fit_bin_centers[-1], 1000)
    cobolt_cal_bs_fit: np.ndarray = exp_2gaus(cobolt_cal_bs_fit_x, *cobolt_cal_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=cobolt_spec_bs, histtype="step", label="$^{60}$Co - background")
    ax.plot(cobolt_cal_bs_fit_x, cobolt_cal_bs_fit, color="black", label="Fit")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Energy [keV]")
    ax.legend(loc="upper right")

    fig.savefig("NaI_cobolt_cal.png", dpi=400, bbox_inches="tight")

    # Background peak resolution

    background_cal_fit_range: tuple[int, int] = (int((1300 - bin_low_edges_cal[0]) / bin_width), int((1600 - bin_low_edges_cal[0]) / bin_width))
    background_cal_fit_bin_centers: np.ndarray = bin_low_edges_cal[background_cal_fit_range[0] : background_cal_fit_range[1]] + bin_width / 2

    background_cal_fit_spec: np.ndarray = background_spec[background_cal_fit_range[0] : background_cal_fit_range[1]]
    background_cal_fit_spec_err: np.ndarray = background_spec_err[background_cal_fit_range[0]: background_cal_fit_range[1]]

    background_cal_fit_spec_nonzero: np.ndarray = background_cal_fit_spec.nonzero()[0]

    background_cal_fit_spec = background_cal_fit_spec[background_cal_fit_spec_nonzero]
    background_cal_fit_spec_err = background_cal_fit_spec_err[background_cal_fit_spec_nonzero]
    background_cal_fit_bin_centers = background_cal_fit_bin_centers[background_cal_fit_spec_nonzero]

    background_cal_fit_guess: list[float] = [7e3, 0.005, 20, 1460, 20]

    try:
        background_cal_p_opt, background_cal_p_cov = so.curve_fit(exp_gaus, background_cal_fit_bin_centers, background_cal_fit_spec, background_cal_fit_guess, 
                                                          background_cal_fit_spec_err, absolute_sigma=True)
        background_cal_p_err: np.ndarray = np.sqrt(np.diag(background_cal_p_cov))
    except:
        print("background calibrated fit failed")
        background_cal_p_opt = np.asarray(background_cal_fit_guess)
        background_cal_p_err = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

    background_resolution: float = 100 * np.sqrt(8 * np.log(2)) * background_cal_p_opt[4] / background_cal_p_opt[3]
    background_resolution_err: float = 100 * np.sqrt(8 * np.log(2)) * np.sqrt( np.power(background_cal_p_err[4] / background_cal_p_opt[3], 2)
                                                 + np.power(background_cal_p_opt[4] / background_cal_p_opt[3]**2 * background_cal_p_err[3], 2))

    print(f"40K peak resolution (1461 keV):  {background_resolution : .3f} +/- {background_resolution_err : .3f}%")

    background_cal_fit_x: np.ndarray = np.linspace(background_cal_fit_bin_centers[0], background_cal_fit_bin_centers[-1], 1000)
    background_cal_fit: np.ndarray = exp_gaus(background_cal_fit_x, *background_cal_p_opt)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(bin_low_edges_cal, nbins, range=(xmin, xmax), weights=background_spec, histtype="step", label="Background")
    ax.plot(background_cal_fit_x, background_cal_fit, color="black", label="Fit")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Energy [keV]")
    ax.legend(loc="upper right")

    fig.savefig("NaI_background_cal.png", dpi=400, bbox_inches="tight")

    return

if __name__ == "__main__":
    main()