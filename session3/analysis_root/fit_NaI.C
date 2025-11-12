#include "Riostream.h"
#include "Math/IFunction.h"
void fit_NaI() {
    ifstream in;
    in.open(Form("data_file_example.txt"));
    Double_t x,y,z;
    Int_t nlines = 0;
    TFile *f = new TFile("output.root","RECREATE");
    TCanvas *c1 = new TCanvas("NaI_energy","NaI_energy",900,700);
    c1->ToggleEventStatus();
    // Calibration coefficients
    float offset=6.644839,slope=1.220401;//enter calibration coefficients we obtained during the experiment (eyeball technique). They are written at the end of the .Spe files OR you can do calibration offline and can be more precise than eyeball calibration. 
    
    // The x-axis (in units of keV), so that E0 is the energy in the
    // middle of the first bin and the maximum energy is in the middle of
    // the last bin. I am assuming there are 8191 bins, but this should be
    // checked for your input file.
    float xmin=offset-slope/2,xmax=offset+slope/2+2048*slope; ///change the channel number from 2048 to 1024 if it is the case for your experiment. 1- Check that from the number of entries in the .Spe files. 2- Or it is written right before the data numbers start in the .Spe files.
   
    TH1F *hist = new TH1F("hist","Energy spectrum;E (keV);Events",2048,xmin,xmax); ///change the channel number from 2048 to 1024 .. the same explanation above is valid also here.
    while (1) {in >> x >> y >> z;
        if (!in.good()) break;
        nlines++;
        // Since the data are already histogrammed we simply set the
        // bin content and bin error of each histogram bin.
        hist->SetBinContent(nlines,(y-z));// subtarction of natural background
        hist->SetBinError(nlines,sqrt(y+z));
    }
    printf(" found %d points\n",nlines);
    in.close();
    
    hist->SetMarkerStyle(20);
    hist->SetMarkerColor(kBlue);
    hist->SetMarkerSize(0.5);
    hist->SetLineColor(kBlue);
    

    Double_t par[5];
    float frmin=600,frmax=720; // Fit range ///change fit range if it is needed.
    TF1 *gspol1 = new TF1("gspol1","gausn(0) + pol1(3)",frmin,frmax);
   //When you are fitting the energy peak, consider changing initial values written bellow for your case if it is required. If the fit with these initial values is not good.
    gspol1->GetChisquare();
    gspol1->SetParameter(0, 1.4e+05); //Norm
    gspol1->SetParameter(1, 662);  // mu
    gspol1->SetParameter(2, 21.00); //sigma
    gspol1->SetParameter(3, 2500);
    gspol1->SetParameter(4, -3.290);
    
    //parameter names for gaussian function
    gspol1->SetParName(0, "norm");
    gspol1->SetParName(1, "mu");
    gspol1->SetParName(2, "sigma");
    //parameter names for polynomial function a + bx.
    gspol1->SetParName(3, "a");  // a (offset)
    gspol1->SetParName(4, "b");  // b (slope)
   
   // hist->Fit(gspol1,"WMER+"); //"WMER" "R+"
   // gspol1->GetParameters(&par[0]);
    
    TF1 *fbkg = new TF1("fbkg","pol1",frmin-50,frmax+50);
    c1->cd(2);
    hist->Draw("E1 X0");
    hist->Fit(gspol1,"PMER+");
    gspol1->GetParameters(&par[0]);
    // Superimpose the background component
    fbkg->SetLineColor(kGreen);
    fbkg->SetParameter(0, par[3]);
    fbkg->SetParameter(1, par[4]);
    fbkg->Draw("SAME");
    gspol1->Draw("SAME");
    cout << par[3] << endl;

    gStyle->SetOptFit();
    
    
    f->Write();
    
}
