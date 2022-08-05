from sys import prefix
import ROOT
import numpy as np
import pickle
import os
from XGBoost_dataprep import Variables
 
 
def load_data_files(signal_filename, background_filename):
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("Events", signal_filename).AsNumpy()
    data_bkg = ROOT.RDataFrame("Events", background_filename).AsNumpy()
 
    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in Variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in Variables]).T
    x = np.vstack([x_sig, x_bkg])
 
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
 
    # Compute weights balancing both classes
    num_all = num_sig + num_bkg
    w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])
 
    return x, y, w

def load_all_data(train):

    if train:
        prefix = "XGtrain_"
    else:
        prefix = "XGtest_"
        
    files = {}
    for file in os.listdir("XGBOOST/"):
        if file.startswith(prefix):
            ind_1 = file.find(prefix) +len(prefix)

            if file.endswith("signal"):
                ind_2 = file.find("signal")
                num = int(file[ind_1:ind_2])
                
                if files[num]:
                    files[num][0] = file
                else:
                    files[num] = [0 for i in range(2)]
                    files[num][0] = file

            if file.endswith("background"):
                ind_2 = file.find("background")
                num = int(file[ind_1:ind_2])
                if files[num]:
                    files[num][1] = file
                else:
                    files[num] = [0 for i in range(2)]
                    files[num][1] = file

    inter_one = True
    x = np.array([])
    y = np.array([])
    w = np.array([])

    # Load data
    for num in files.keys():
        if iter_one:
             x, y, w = load_data(files[num][0], files[num][1])
             iter_one = False
        
        x_temp, y_temp, w_temp = load_data(files[num][0], files[num][1])
        x = np.concatenate(x,x_temp)
        y = np.concatenate(y,y_temp)
        w = np.concatenate(w,w_temp)

    return x,y,w
 
def XGBoost_Train(pars,name):

    x,y,w = load_all_data(True)
    # Fit xgboost model
    from xgboost import XGBClassifier
    maxdepth = pars[0]
    nestimators = pars[1]

    bdt = XGBClassifier(max_depth=maxdepth, n_estimators=nestimators)
    bdt.fit(x, y, w)
 
    # Save model in TMVA format
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, name, "XGBOOST/"+name+".root")