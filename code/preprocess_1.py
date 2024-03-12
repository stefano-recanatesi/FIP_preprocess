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

#%
# -----------------------------------------------------------------------------------
# LOAD, PREPROCESS AND SAVE THE NEURAL DATA
def main():    

    # NEW(Previously a hidden function)--------------------
    # parse the folder name for the data
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--folder', help='folder name', required=True)
    args = parser.parse_args()

    # define the path to the data
    data_path = os.path.abspath("/data")
    folder = args.folder
    sub_folder = r"FIP"
    folder_path = os.path.join(data_path,folder,sub_folder)
    AnalDir = os.path.abspath(str(folder_path))
    print("AnalDir:", AnalDir)
    # ----------------------------------------------
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

    # define 2 dataframes to hold the processed data and some preprocessing parameters
    neural_data_df = pd.DataFrame()
    params_df = pd.DataFrame()

    # load the data
    Data_1, Data_2, PMts = dl.acquire_neural_data(AnalDir)
    datasets = Data_1, Data_2

    print("starting preprocessing now")

    # preprocess the data and store it in a df
    for i, data in enumerate(datasets):
        # make labels for dataframe
        data_labels = ['Ctrl{}'.format(i+1), 'G{}'.format(i+1), 'R{}'.format(i+1)]
        # preprocessing step + storing in dataframe
        for j, label in enumerate(data_labels):
            neural_data_df[label], params_df[label] = pp.tc_preprocess(data[j], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile) # NEW added output 

    # Save the data (df_f traces now in a single DF)
    results_folder = dl.define_resultsDir(folder)

    # df_f presented in percentage form
    neural_data_df = neural_data_df*100
    neural_data_df = neural_data_df[['G1','R1','G2','R2']]

    neural_data_df.to_pickle(results_folder + os.sep + 'neural_data.pkl')
    params_df.to_pickle(results_folder + os.sep + 'neural_data_params.pkl')

#%
# --------------------------------------------------------------------
# if this is the main script being run, then execute the following code
if __name__ == "__main__":
    main()

#%% Dataframe repackaging snippet
# df_fip = pd.DataFrame()
# for col in neural_data_df.columns:
#     df_col = pd.DataFrame({'timestamp':np.nan,'fip':neural_data_df[col],'channel':col[0],'area':col[1], 'preprocess':'None'})
#     df_fip = pd.concat([df_fip, df_col], axis=0)