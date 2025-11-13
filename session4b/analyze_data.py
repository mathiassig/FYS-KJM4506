import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, chisquare, exponweib, gamma
from scipy.optimize import curve_fit
import scipy.special as sci

class ManageData:
    def __init__(self, filename, create_fig=True, calibration_coeffs = None):
        self.filename = filename
        self.calibration = calibration_coeffs # found manually, passed as argument
        self._read_spe()
        self.get_fit_called = False
        self.calibrate_background = False
        self.calibrate_calibration = False
        self.calibrate_scale = False
        self.model = None
        if create_fig: self.fig, self.ax = plt.subplots(figsize=(10, 8))
    
    def _read_spe(self):
        """
        Extract channel and signal data from .spe files.

        Returns
        -------
        self.signal : numpy.ndarray
            Signal data.

        self.channels : numpy.ndarray
            Channel data.
        """
        self.first_channel, self.last_channel = np.genfromtxt(self.filename,
            skip_header=11, max_rows=1, dtype=int)
        self.signal = np.genfromtxt(self.filename, skip_header=12,
            max_rows=self.last_channel+1, dtype=float)
        if self.calibration is None: # if no calibration coeffs are passed, read them from .Spe files
            self.calibration = np.genfromtxt(self.filename,
                skip_header=11+self.last_channel+10+2, max_rows=1, dtype=float, 
            usecols=[0,1,2])
        self.channels = np.arange(self.first_channel, self.last_channel+1, 1, dtype=float)        

    def calibrate_data(self, calibration=False, background=None, scale=False):
        """
        Calibrate data set.

        Parameters
        ----------
        calibration : boolean
            Toggle channel calibration on / off.  Use calibration data
            for intercept and slope of the channels from the .Spe file.

        background : None, numpy.ndarray
            Array of same dimensions as self.signal containing
            background noise to be subtracted.  Defaults to None where
            no background is subtracted.

        scale : boolean
            Toggle scaling (normalization) of the data on / off.
            Defaults to False.
        """
        if background is not None:
            self.calibrate_background = True
            self.signal -= background
        
        self.max_idx = np.argmax(self.signal)

        if calibration:
            """
            Calibrate the channels using a first degree polynomial: 
            channels = a + b*channels.
            """
            self.calibrate_calibration = True
            a, b, _ = self.calibration
            self.channels = a + b*self.channels

        if scale:
            self.calibrate_scale = True
            self.signal /= np.trapz(self.signal)

    def plot_data(self, label=None, show_plot=True):
        """
        Plot the signal data as a function of channels.

        Parameters
        ----------
        label : None, string
            Label of the data.  Defaults to None where filename of .spe
            file is used.

        show_plot : boolean
            Toggle plt.show() on / off.  Toggle to off if more data is
            to be plotted before show.
        """

        if label is None: label = self.filename
        
        self.ax.plot(self.channels[self.first_channel:self.last_channel], self.signal[self.first_channel:self.last_channel], label=label) # endrer for å få den til å plotte alle kanalene
        idx = np.argmin(np.abs(self.signal[0:500] - 184.3e3))
        # plt.plot(self.channels[idx], self.signal[idx], "r.")
        # self.ax.plot(self.channels[100:230], self.signal[100:230])
        self._title_label_legend()
        if show_plot: plt.show()

    def get_fit(self, curve_start, curve_stop, init_guess=None, plot_fit=False,
            show_plot=True, model="normal"):
        """
        Find fit to data using scipy.optimize.curve_fit.

        Parameters
        ----------
        curve_start : int
            Index of signal and channel array where the curve to be
            fitted starts.

        curve_stop : int
            Index of signal and channel array where the curve to be
            fitted stops.

        init_guess : None, list
            A list of initial guesses containing [amplitude, mu, sigma]
            for a normal distribution.  Defaults to None where no
            fitting is performed, but the signal and channel arrays are
            plottet with the input curve_start and curve_stop indices.

        plot_flit : boolean
            Toggle view if the fit on / off.

        show_plot : boolean
            Toggle plt.show() on / off.  Toggle to off if more data is
            to be plotted before show.

        model : string
            Choose between 'normal' and 'chi2' / 'chisquared'.        
        """
        
        self.get_fit_called = True
        self.model = model
        signal_slice = self.signal[curve_start:curve_stop]
        channel_slice = self.channels[curve_start:curve_stop]

        if init_guess is None:
            """
            If no initial guess is given, assume that plot must be
            studied to find suitable values.
            """
            self.ax.plot(channel_slice, signal_slice, label='no fit, only slice')
            self.ax.legend()
            plt.show()
        else:
            if self.model == "normal":
                """
                Model function is the normal distribution.
                """
                if len(init_guess) != (3):
                    print("Please supply initial guess for amplitude, expectation value and variance.")
                    return
                
                self.popt, self.pcov = curve_fit(f=gauss_wrapper, xdata=channel_slice,
                    ydata=signal_slice, p0=init_guess, method="lm")
                signal_slice_model = gauss_wrapper(channel_slice, *self.popt)
            
            elif (self.model == "chisquare") or (self.model == "chi2"):
                """
                Model function is the chi square distribution.
                """
                if len(init_guess) != 2:
                    print("Please supply initial guess for amplitude and dof.")
                    return
                
                self.popt, self.pcov = curve_fit(f=chi_wrapper, xdata=channel_slice,
                    ydata=signal_slice, p0=init_guess)
                signal_slice_model = chi_wrapper(channel_slice, *self.popt)
            
            elif self.model == "weibull":
                self.popt, self.pcov = curve_fit(f=weibull_wrapper, xdata=channel_slice,
                    ydata=signal_slice, p0=init_guess)
                signal_slice_model = weibull_wrapper(channel_slice, *self.popt)

            elif self.model == "gamma":
                self.popt, self.pcov = curve_fit(f=gamma_wrapper, xdata=channel_slice,
                    ydata=signal_slice, p0=init_guess)

                signal_slice_model = gamma_wrapper(channel_slice, *self.popt)

            else:
                msg = "Please choose model = 'normal' or 'chisquare' or 'weibull' or 'gamma'."
                raise ValueError(msg)
                
            #self.chisq, self.p = chisquare(f_obs=signal_slice, f_exp=signal_slice_model) denne lager bare kødd

        if plot_fit:
            if init_guess is None:
                msg = "No initial guess is given!"
                raise ValueError(msg)
            
            # har redigert denne
            if self.model == "normal":
                A_err, mu_err, sigma_err= np.sqrt(np.diag(self.pcov)) # , slope_err, intercept_err 
                tekst = f'fit:\n' + f'$\mu = {self.popt[1]:.1f} \pm {mu_err:.1f}$,\n'
                tekst += f'$\sigma = {self.popt[2]:.1f} \pm {sigma_err:.1f}$\n'
                tekst+= f'$A = {self.popt[0]} \pm {A_err:.1f}$\n'
                print(tekst)
                # label += f'intercept = {self.popt[4]:.1f} $\pm$ {intercept_err:.1f}\n'
                # label += f'slope = {self.popt[3]:.1f} $\pm$ {slope_err:.1f}\n'
                # label += f'$\chi^2 = {self.chisq:.1f} / {curve_stop - curve_start}$\n'
            
            elif (self.model == "chisquare") or (self.model == "chi2"):
                label = f'$fit: dof = {self.popt[1]:.1f}$'
            
            elif self.model == "weibull":
                label = f'fit: {self.model}\n'
                label += f'mean =  {np.average(a=channel_slice, weights=signal_slice_model):.1f}\n'
                label += f'$\chi^2 = {self.chisq:.1f} / {curve_stop - curve_start}$\n'
                label += f'ampl: {self.popt[0]}\n'
                label += f"mean: {self.popt[1]*sci.gamma(1 + 1/self.popt[2])}\n"
                label += f'$\lambda = {self.popt[1]:.1f}$\n'
                label += f'$k = {self.popt[2]:.1f}$\n'
                label += f'$loc = {self.popt[3]:.1f}$\n'
                label += f'$scale = {self.popt[4]:.1f}$\n'
                label += f'slope = {self.popt[5]:.1f}\n'
                label += f'intercept = {self.popt[6]:.1f}\n'
            
            elif self.model == "gamma":
                A_err, k_err, loc_err, scale_err, slope_err, intercept_err = np.sqrt(np.diag(self.pcov))
                label = f'fit: {self.model}\n'
                label += f'mean =  {np.average(a=channel_slice, weights=signal_slice_model):.1f}\n'
                label += f'$\chi^2 = {self.chisq:.1f} / {curve_stop - curve_start}$\n'
                label += f'$k = {self.popt[1]:.1f} \pm {k_err:.1f}$\n'
                label += f'slope = {self.popt[4]:.1f} $\pm$ {slope_err:.1f}\n'
                label += f'intercept = {self.popt[5]:.1f} $\pm$ {intercept_err:.1f}\n'
            

            # self.ax.plot(channel_slice, signal_slice, label='data')
            self.ax.plot(channel_slice, signal_slice_model)
            self._title_label_legend()
            if show_plot: plt.show()

    def _title_label_legend(self):
        self.ax.set_title(f"subtracted background: {self.calibrate_background}, channel calibration: {self.calibrate_calibration}, scaled data: {self.calibrate_scale}")
        if self.calibrate_calibration: # if spectrum is calibrated
            self.ax.set_xlabel("Energy [KeV]", fontsize=15)
        else: # if not calibrated, return channel number
            self.ax.set_xlabel("Channel", fontsize=15)
        self.ax.set_ylabel("# events", fontsize=15)
        self.ax.tick_params(labelsize=15)
        self.ax.legend(fontsize=12)

    def resolution(self, E0):
        """
        Find the resolution by the formula r = 100*FWHM/E0.

        Parameters
        ----------
        E0 : int, float
            The channel value of the signal peak.

        Returns
        -------
        : None
            If get_fit method is not called first.

        : float
            The resolution.
        """
        if not self.get_fit_called:
            print("Resolution cannot be found before fit is found.")
            return

        if self.model != "normal":
            print(f"Resolution not implemented for {self.model}.")
            return
        
        _, _, sigma, _, _ = self.popt
        self.FWHM = 2*np.sqrt(2*np.log(2))*sigma
        self.resolution = 100*self.FWHM/E0

        return self.resolution

def chi_wrapper(x, A, df):
    return A*chi2.pdf(df=df, x=x)

def gauss_wrapper(x, A, mu, sigma):
    """
    A wrapper function for the probability distribution function for the
    normal distribution.  Use to pass as argument to curve_fit.

    Parameters
    ----------
    x : numpy.ndarray
        Slice of channel values.

    A : int, float
        Amplitude of curve.
    
    mu : int, float
        Expectation value.

    sigma : int, float
        Variance.

    slope : int, float
        Polynomial slope.

    intercept : int, float
        Polynomial intercept.    

    Returns
    -------
    Wrapped function.
    """
    return A*norm.pdf(x, mu, sigma)

def weibull_wrapper(x, A, lambd, k, loc, scale, slope, intercept):
    rv = exponweib(lambd, k, loc, scale)
    return A*rv.pdf(x) + slope*x + intercept

def gamma_wrapper(x, A, k, loc, scale, slope, intercept):
    return A*gamma.pdf(x, k, loc, scale) + slope*x + intercept

if __name__ == "__main__":
    # switches
    distance_plot = False
    sources_plot = True
    shapingtime_plot = False
    calibration_plot_germanium = False
    calibration_switch = False
    fit_switch = False # whether to fit peaks with gaussian or not
    #calibration_coeffs = [-445.42857142857133,4.365079365079365,0] # last one is dummy, order is reversed because of how class uses them
    calibration_coeffs = None # the ones found using Barium where absolutely terrible, useless

    Ba_gamma_1 = 81.0 # keV
    Ba_gamma_2 = 356 # keV
    Cs_gamma = 661.659  # keV.
    Co_gamma_1 = 1173.2   # keV.
    Co_gamma_2 = 1332.5   # keV.
    # model = "gamma"

    # find background for germanium
    background = ManageData("HPGe_background.Spe",create_fig=False)

    if calibration_plot_germanium:
        #calibration = ManageData("calibration.Spe") # use this to find calibration parameters
        calibration = ManageData("HPGe_calibration.Spe") # use this to find calibration parameters
        calibration.calibrate_data(calibration=False, background=background.signal, scale=False)
        calibration.plot_data(show_plot=True, label=r"$^Calibration data")
        # width =16
        # max = 134-10 # bin number of peak, since this is pre-calibrated spectrum
        # calibration.get_fit(
        #     curve_start = max-width,
        #     curve_stop = max+width,
        #     init_guess = [6000, max, width],   # Amplitude, mean, variance. Guess for normal distribution 
        #     plot_fit = True,
        #     show_plot = False, # are plotting more peaks
        #     model = "normal"
        # ) 
        # calibration2 = ManageData("133Ba_5cm.Spe") # use this to find calibration parameters
        # calibration2.calibrate_data(calibration=False, background=background.signal, scale=False)
        # calibration2.plot_data(show_plot=False, label=r"$^Calibration data")
        # width = 40
        # max = 493-10 # bin number of peak, since this is pre-calibrated spectrum
        # calibration2.get_fit(
        #     curve_start = max-width,
        #     curve_stop = max+width,
        #     init_guess = [2000, max, width],  # Amplitude, mean, variance, slope, intercept. Guess for normal distribution with slope.
        #     plot_fit = True,
        #     show_plot = True,
        #     model = "normal"
        # ) 

    # Plot 137Cs, normal fit of the peaks, for distances 5-20 cm
    if distance_plot:
        distances = ["5cm","10cm","15cm","20cm"]
        for distance in distances:
            Cs137 = ManageData(f"137Cs_{distance}.Spe",calibration_coeffs=calibration_coeffs)
            Cs137.calibrate_data(calibration=True, background=background.signal, scale=False)
            Cs137.plot_data(show_plot=False, label=r"$^{137}$Cs data")
            adjust = -50
            Cs137.get_fit(
                curve_start = 450 - adjust+300,
                curve_stop = 630 + adjust+380,
                init_guess = [10000, Cs_gamma, 20],   # Amplitude, mean, variance Guess for normal distribution 
                plot_fit = True,
                show_plot = True,
                model = "normal"
            )
    if sources_plot:
        sources = ["133Ba","137Cs_8micros","60Co"]
        for source in sources:
            data = ManageData(f"HPGe_{source}.Spe",calibration_coeffs=calibration_coeffs)
            data.calibrate_data(calibration=calibration_switch, background=background.signal, scale=False)
            if source=="133Ba":
                if fit_switch:
                    data.plot_data(show_plot=False, label=rf"{source} data")
                    adjust = -50
                    data.get_fit(
                        curve_start = 50 - adjust,
                        curve_stop = 200 + adjust,
                        init_guess = [10000, Ba_gamma_1, 20],   # Amplitude, mean, variance Guess for normal distribution
                        plot_fit = True,
                        show_plot = False, # plotting more peaks
                        model = "normal"
                    ) 
                    data2 = ManageData(f"HPGe_{source}.Spe",calibration_coeffs=calibration_coeffs)
                    data2.calibrate_data(calibration=calibration_switch, background=background.signal, scale=False)
                    data2.plot_data(show_plot=False, label=rf"{source} data")
                    adjust = -50
                    data2.get_fit(
                        curve_start = 370 - adjust,
                        curve_stop = 600 + adjust,
                        init_guess = [10000, Ba_gamma_2, 20],   # Amplitude, mean, variance. Guess for normal distribution 
                        plot_fit = True,
                        show_plot = True,
                        model = "normal"
                    )
                else:
                    data.plot_data(show_plot=True, label=rf"{source} data")
            elif source=="60Co":
                if fit_switch:
                    data.plot_data(show_plot=False, label=rf"{source} data")
                    adjust = -50
                    data.get_fit(
                    curve_start = 1250 - adjust+120,
                    curve_stop = 1550 + adjust+120,
                    init_guess = [100, Co_gamma_1, 20],   # Amplitude, mean, variance Guess for normal distribution 
                        plot_fit = True,
                        show_plot = False, # are plotting more peaks
                        model = "normal"
                    ) 
                    data2 = ManageData(f"HPGe_{source}.Spe",calibration_coeffs=calibration_coeffs)
                    data2.calibrate_data(calibration=calibration_switch, background=background.signal, scale=False)
                    data2.plot_data(show_plot=False, label=rf"{source} data")
                    adjust = -50
                    data2.get_fit(
                    curve_start = 1410 - adjust+150,
                    curve_stop = 1700 + adjust+150,
                    init_guess = [1000, Co_gamma_2, 20],   # Amplitude, mean, variance, . Guess for normal distribution.
                        plot_fit = True,
                        show_plot = True,
                        model = "normal"
                    )
                else:
                    data.plot_data(show_plot=True, label=rf"{source} data")
            elif source == "137Cs_8micros":
                if fit_switch:
                    data.plot_data(show_plot=False, label=r"$^{137}$Cs data")
                    adjust = -50
                    data.get_fit(
                    curve_start = 450 - adjust+300,
                    curve_stop = 630 + adjust+380,
                    init_guess = [10000, Cs_gamma, 20],   # Amplitude, mean, variance Guess for normal distribution 
                    plot_fit = True,
                    show_plot = True,
                    model = "normal"
                    )
                else:
                    data.plot_data(show_plot=True, label=r"$^{137}$Cs data")
    # # Plot other peaks in the 137Cs file, normal
    # Cs137_spike = ManageData("137Cs.Spe")
    # Cs137_spike.calibrate_data(calibration=True, background=background.signal, scale=False)
    # Cs137_spike.plot_data(show_plot=False, label=r"$^{137}$Cs data")
    
    # adjust = -50
    # Cs137_spike.get_fit(
    #     curve_start = 100 - adjust,
    #     curve_stop = 150 + adjust,
    #     init_guess = [10000, 10, 20, -1e-3, 0],   # Amplitude, mean, variance, slope, intercept. Guess for normal distribution with slope.
    #     plot_fit = True,
    #     show_plot = True,
    #     model = "normal"
    # )


    # Other types of fits
    # Cs137 = ManageData("137Cs.Spe")
    # Cs137.calibrate_data(calibration=True, background=background.signal, scale=False)
    # Cs137.plot_data(show_plot=False, label=r"$^{137}$Cs data")

    # Cs137.get_fit(
    #     curve_start = 120,
    #     curve_stop = 190,
    #     init_guess = [2000, 1, 180, 1, 0, 0],  # Amplitude, gamma parameter, loc, scale, slope, intercept.
    #     plot_fit = True,
    #     show_plot = True,
    #     model = "gamma"
    # )

    

    # elif model == "chisquare":
    #     Cs137_init_guess = [10000, 650]   # Amplitude, mean, variance. Guess for chi square distribution.
    
    # elif model == "weibull":
    #     Cs137_init_guess = [2000, 1, 1.5, 155, 2, 0, 0]   # Amplitude, lambda, k, loc, scale, slope, intercept.
    #     Cs137.get_fit(
    #         curve_start = 120,
    #         curve_stop = 190,
    #         init_guess = Cs137_init_guess,
    #         plot_fit = True,
    #         show_plot = True,
    #         model = model
    #     )
    # [2000, 184, 5, 0, 0]
    # Cs137.get_fit(curve_start=120, curve_stop=190)
    # print(f"scope: {Cs137.channels[450-adjust]:.0f}:{Cs137.channels[630+adjust]:.0f}")
    # print(f"resolution : {Cs137.resolution(E0=Cs_gamma)}")
    # print(f"chisq: {Cs137.chisq}, pval: {Cs137.p}, res: {Cs137.resolution}")
    # print(f"dof?: {(630+adjust) - (450-adjust)}")
