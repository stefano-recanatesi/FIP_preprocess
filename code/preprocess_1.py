""" Modularized preprocessing script. 
    Outputs preprocessed traces for the FIP data.
"""


#%% clear all
from IPython import get_ipython
#get_ipython().magic("reset -sf")

#%%
import os
import csv
import numpy as  np
import pandas as pd
import pylab as pl
import glob
import pickle
import argparse
from pathlib import Path


import Preprocessing_library as pp
import data_acquisition_library as dl

# -------------------------------------------------------------------------------------

# LOAD, PREPROCESS AND SAVE THE NEURAL DATA
def main():

    print("Acquiring data now")
    
    # define variables
    nFibers = 2
    nColor = 3
    sampling_rate = 20 #individual channel (not total)
    nFrame2cut = 100  #crop initial n frames
    b_percentile = 0.70 #To calculare F0, median of bottom x%
    BiExpFitIni = [1,1e-3,1,1e-3,1]  #currently not used (instead fitted with 4th-polynomial)
    kernelSize = 1 #median filter
    degree = 4 #polyfit

    # dataframe to hold the processed data
    neural_data_df = pd.DataFrame()

    # load the data
    folder, Analysis_dir = dl.define_AnalDir()
    Data_1, Data_2, PMts = dl.acquire_neural_data(Analysis_dir)
    datasets = Data_1, Data_2

    print("starting preprocessing now")
  
    # preprocess the data and store it in a df
    for i, data in enumerate(datasets):
        # make labels for dataframe
        data_labels = ['Ctrl{}'.format(i+1), 'G{}'.format(i+1), 'R{}'.format(i+1)]
        # preprocessing step + storing in dataframe
        for j, label in enumerate(data_labels):
            neural_data_df[label] = pp.tc_preprocess(data[j], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)


    # Save the data (df_f traces now in a single DF)
    results_folder = dl.define_resultsDir(folder)

    # df_f presented in percentage form
    neural_data_df = neural_data_df*100
    neural_data_df = neural_data_df[['G1','R1','G2','R2']]

    neural_data_df.to_pickle(results_folder + os.sep + 'neural_data.pkl')

# --------------------------------------------------------------------
# if this is the main script being run, then execute the following code
if __name__ == "__main__":
    main()
