# use Python interface instead, since we couldn't get the c++ file to compile
import ROOT
import numpy as np
import ctypes
from array import array
##################################################################################
#################### Import data ####################################
sourcename=input("Enter sourcename: (ex:239Pu)")
data = np.loadtxt(f'data/{sourcename}.txt',delimiter = ";",dtype=str)#load the data



#Some useful settings for ROOT
#We want the statistics box to be displayed and the fit information to appear in the box
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat()
ROOT.gStyle.SetOptFit(11111111)


#Set up the arrays with time `x_val`, counts `y_val`, and uncertainties on the counts `y_errs`.
#Also set up arrays for the log of the counts `y_log` and the error on the log of the counts `y_logerrs`. 
# You should enter your own data.
# #  Don't enter any data for bins where the background subtraction gives less than zero counts, or the linear fit below won't work.
if sourcename == "137Ba": # ad hoc løsning
    B = (data[1,2]).astype(float)/(data[1,3]).astype(float)*10 # Estimated constant background counts per bin (10 second bin size)
elif sourcename == "234Pa":
    B=((data[1:3,2]).astype(float)).mean()/(data[1,3]).astype(float)*10 # Estimated constant background counts per bin (10 second bin size)
else:
    B=0
x_val = array('d',(data[1:,0]).astype(float))
y_val = array('d',(data[1:,1]).astype(float))
y_errs = array('d',(data[1:,1]).astype(float)**0.5) # Bruker at for Poisson-fordeling er std = sqrt(N)

y_log = np.log(y_val)
y_logerrs = 1/np.sqrt(y_val)

# # Create one Monte Carlo toy dataset
# # Model is constant background plus an exponential decay. **Comment this out to avoid over-writing your own data.**

# t = np.arange(0,400.1,20)  # Array of times of measurements
# B = 30                     # Constant background rate
# N0 = 260                   # Number of 234Pa at time 0
# thalf0 = 69.54             # Nominal half-life of 234 Pa
# tau0 = thalf0/np.log(2.0)
# F = B + N0*np.exp(-t/tau0) # Complete model
# data = np.zeros(len(F))    # Empty data array

# rng = np.random.default_rng() # For the Poisson random number generator
# # Generate data in each bin with Poisson statistics, but suppress the bins
# # starting with the first bin with less than 0 counts after background subtraction.
# for i in range(len(F)):
#     d = rng.poisson(F[i])
#     if d-B > 0:
#         data[i] = d
#     else:
#         data = data[0:i-1]
#         break

# # Put the dataset into arrays that ROOT can digest
# x_val = array('d',t[1:len(data)])
# y_val = array('d',data-B)
# y_errs= array('d',np.sqrt(data))

# # Prepare to analyse the log of the background-subtracted data
# y_log = np.log(y_val)
# y_logerrs = np.sqrt(data)/y_val

#####################################################################

# Perform an exponential fit to the data and display the results
canvas1 = ROOT.TCanvas("canvas1","Raw data",800,600);

graph1 = ROOT.TGraphErrors(len(x_val),x_val,y_val,ROOT.nullptr,y_errs)
graph1.Draw("APE")

graph1.SetTitle("Exponential decay curve;time (sec);counts/sec");
graph1.SetMarkerStyle(20);
graph1.SetLineColor(ROOT.kBlack);

# define the fit function
fexp = ROOT.TF1("fexp","expo(0)",20.0,300.0);
fexp.SetLineColor(ROOT.kRed);
fexp.SetLineStyle(1);
fexp.SetParNames("log(A_0)","Decay constant");

# Fit it to the graph and draw it
gf = graph1.Fit("fexp","S");
fexp.Draw("Same");

# Display the plot
canvas1.Draw()
# Save the plot
canvas1.SaveAs(f"figures/{sourcename}_expfit.png")
# Calculate the half-life from the fitted decay constant (lambda or 1/lifetime)
thalf = -np.log(2)/gf.Value(1)
dthalf = np.log(2)*gf.Error(1)/gf.Value(1)**2

print('Half-life =',"{:.1f}".format(thalf),'+-',"{:.1f}".format(dthalf),'s')
print('')

##############################################################################
# Perform a linear fit to the natural logarithm of the data and display the results
canvas2 = ROOT.TCanvas("canvas2","Linearized",800,600);

graph2 = ROOT.TGraphErrors(15,x_val,y_log,ROOT.nullptr,y_logerrs)
graph2.Draw("APE")

graph2.SetTitle("Exponential decay curve;time (sec);log(counts/sec)");
graph2.SetMarkerStyle(20);
graph2.SetLineColor(ROOT.kBlack);

# define the linear fit function
flin = ROOT.TF1("flin","pol1",20,300);
flin.SetLineColor(ROOT.kBlue);
flin.SetLineStyle(1);
flin.SetParNames("Intercept","Slope");

# Fit it to the graph and draw it
gf = graph2.Fit("flin","S");
flin.Draw("Same");

# Display the plot
canvas2.Draw()
# Save the plot
canvas2.SaveAs(f"figures/{sourcename}_linfit.png")

# Calculate the half-life from the fitted decay constant (lambda or 1/lifetime)
thalf = -np.log(2)/gf.Value(1)
dthalf = np.log(2)*gf.Error(1)/gf.Value(1)**2

print('Half-life =',"{:.1f}".format(thalf),'+-',"{:.1f}".format(dthalf),'s')
print('')

