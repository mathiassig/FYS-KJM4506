#include "TCanvas.h"
#include "TROOT.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TLegend.h"
#include "TArrow.h"
#include "TLatex.h"
#include "TFitResultPtr.h"
#include "TRandom.h"
#include "math.h"
#include <iostream>

// Original version by Eda Sahin (ca 2019)
// Updated by Alex Read 01.11.2022

void plot_errors(){
    gROOT->SetStyle("Plain");
    gStyle->SetOptStat();
    gStyle->SetOptFit(11111111);
    
    const int n_points=21;
    double x_val[n_points]={0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400};
    double data[n_points]={229,180,88,110,110,100,72,52,45,38,38,33,30,38,25,38,33,30,38,25};
    double B = 24; // Background estimate
    
    //
    // Calculate the log of the y values and the corresponding uncertainties
    // Subtract the background B, but don't let net counts become negative.
    // The statistical error is given by the gross number of counts. This
    // assumes that the background uncertainty is negligible with respect to
    // the statistical uncertainty for the decay data.
    //
    // You might want to simply not enter data starting with the first bin with
    // negative net counts.
    //
    double y_val[n_points], y_errs[n_points], y_log[n_points], y_logerrs[n_points];
    for(int i=0;i<n_points;i++) {
      if (data[i] > B) {
	y_val[i] = data[i]-B;
      }
      else
	{
	  y_val[i] = 1;
	}
      y_errs[i] = sqrt(data[i]);
      y_log[i] = log(y_val[i]);
      y_logerrs[i] = sqrt(data[i])/y_val[i];
    }

    // Generate a Monte Carlo toy dataset. You will want to comment this out
    // if you have entered your data above.
    //
    TRandom *r0 = new TRandom();
    r0->SetSeed(0);                  // Random seed based on system time. Comment to get default deterministic seed.
    B = 30;                          // Constant background rate
    double N0 = 260;                 // Number of 234Pa at time 0
    double thalf0 = 69.54;           // Nominal half-life of 234 Pa
    double tau0 = thalf0/log(2.0);
    double F[n_points];
    for (int i=0;i<n_points;i++) {
      F[i] = B + N0*exp(-x_val[i]/tau0); // Complete model
      data[i] = r0->Poisson(F[i]);       // Poisson fluctuations
      if (data[i] > B) {                 // Prevent net counts from being < 0
	y_val[i] = data[i]-B;
      }
      else
	{
	  y_val[i] = 1;
	}
      y_val[i] = data[i]-B;
      y_errs[i] = sqrt(data[i]);
      y_log[i] = log(y_val[i]);
      y_logerrs[i] = sqrt(data[i])/y_val[i];
    }
    // End of Monte Carlo toy generation
    
    TCanvas* mycanvasExp = new TCanvas("canvasExp","Raw",800,600);

    TGraphErrors *graphExp= new TGraphErrors(n_points,x_val,y_val,NULL,y_errs);

    graphExp->SetTitle("Exponential decay curve;time (sec);counts/sec");
    graphExp->SetMarkerStyle(20);
    graphExp->SetLineColor(kBlack);
 
    graphExp->Draw("APE");
    
    //define the fit function
    TF1 *f2 = new TF1("f2","expo(0)",20.0,300.0);
    f2->SetLineColor(kRed);
    f2->SetLineStyle(1);
    f2->SetParNames("log(A_0)","Decay constant");
    // Fit it to the graph and draw it
    TFitResultPtr gf = graphExp->Fit("f2","S");
    f2->Draw("Same");
    graphExp->SetMinimum(0);
    
    mycanvasExp->Print("decay_curve_exp.pdf");  // Save the canvas to a pdf file

    // Calculate the half-life from the fitted decay constant (-lambda or -1/lifetime)
    double thalf = -log(2)/gf->Value(1);
    double dthalf = log(2)*gf->Error(1)/(gf->Value(1)*gf->Value(1));

    std::cout << std::endl << "Half-life = " << setprecision(3) << thalf << "+-" << setprecision(2) << dthalf << " s" << std::endl << std::endl;

    //////////////////------------------------------------------------------------------------///////////////////

    // Define the linear fit function and fit it to log of the data
    // Don't forget to subract the background! We shouldn't be surprised
    // if the fit parameters are nearly the same, since "expo(0)" is
    // interpreted by ROOT to be exp(p[0]+p[1]*x), rather than the
    // perhaps more standard A_0*exp(-lambda*t). In other words,
    // A_0 corresponds to exp(p[0]).

    TCanvas* mycanvasLin = new TCanvas("canvasLin","Linearized",800,600);

    TGraphErrors *graphLin = new TGraphErrors(n_points,x_val,y_log,NULL,y_logerrs);
     
    graphLin->SetTitle("Linearized decay curve;time (sec);log(counts/sec)");
    graphLin->SetMarkerStyle(20);
    graphLin->SetLineColor(kBlack);
    
    graphLin->Draw("APE");//

    TF1 *flin = new TF1("flin","pol1",20,300);
    flin->SetLineColor(kBlue);
    flin->SetLineStyle(1);
    flin->SetParNames("Intercept","Slope");
    gf = graphLin->Fit("flin","S");

    mycanvasLin->Print("decay_curve_lin.pdf"); // Save the canvas to a pdf file

    // Calculate the half-life from the fitted decay constant (-lambda or -1/lifetime)
    thalf = -log(2)/gf->Value(1);
    dthalf = log(2)*gf->Error(1)/(gf->Value(1)*gf->Value(1));

    std::cout << std::endl << "Half-life = " << setprecision(3) << thalf << "+-" << setprecision(2) << dthalf << " s" << std::endl;
}
#ifndef __CINT__
int main(){
    plot_errors();
}
#endif


