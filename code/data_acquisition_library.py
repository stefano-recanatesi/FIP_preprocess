"""This is library of functions to load the data """
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
# ---------------------------------------------------------------------------------------

 
# READ IN THE SESSION NAME + DEFINE THE ANALYSIS DIRECTORY
def define_AnalDir():
    # Create an ArgumentParser for command-line arguments
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--folder', help='folder name', required=True)
    args = parser.parse_args()

    # define the path to the data
    data_path = os.path.abspath("/data")

    # Access the folder using args.folder
    folder = args.folder
    sub_folder = r"FIP"

    # get the path to the folder
    folder_path = os.path.join(data_path,folder,sub_folder)

    # neutral version
    AnalDir = os.path.abspath(str(folder_path))
    print("AnalDir:", AnalDir)

    return folder, AnalDir


# LOAD DATA 
def acquire_neural_data(AnalDir, sampling_rate=20):

    #%% Data Loading -- with FIP data only
    file1 = None
    file2 = None
    file3 = None

    if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
        print('preprocessing Neurophotometrics Data')
        file1  = glob.glob(AnalDir + os.sep + "L415*")[0]
        file2 = glob.glob(AnalDir + os.sep + "L470*")[0]
        file3 = glob.glob(AnalDir + os.sep + "L560*")[0]

        
    elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
        print('preprocessing FIP Data')
        file1  = glob.glob(AnalDir + os.sep + "FIP_DataIso*")[0]
        file2 = glob.glob(AnalDir + os.sep + "FIP_DataG*")[0]
        file3 = glob.glob(AnalDir + os.sep + "FIP_DataR*")[0]
        
    else:
        print('photometry raw data missing; please check the folder specified as AnalDir')


    with open(file1) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data1 = datatemp[1:,:].astype(np.float32)
        #del datatemp
        
    with open(file2) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data2 = datatemp[1:,:].astype(np.float32)
        PMts= data2[:,0] 
        #del datatemp
        
    with open(file3) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data3 = datatemp[1:,:].astype(np.float32)
        #del datatemp
            
    # in case acquisition halted accidentally
    Length = np.amin([len(data1),len(data2),len(data3)])
    data1 = data1[0:Length] 
    data2 = data2[0:Length]
    data3 = data3[0:Length]

    if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
        Data_Fiber1iso = data1[:,8]
        Data_Fiber1G = data2[:,8]
        Data_Fiber1R = data3[:,10]
        
        Data_Fiber2iso = data1[:,9]
        Data_Fiber2G = data2[:,9]
        Data_Fiber2R = data3[:,11]

    #** here, Stefano added a case to cater for acquisition halted unexpectedly
    # check if it makes a difference then ADD or ignore

    elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
        Data_Fiber1iso = data1[:,1]
        Data_Fiber1G = data2[:,1]
        Data_Fiber1R = data3[:,1]
        
        Data_Fiber2iso = data1[:,2]
        Data_Fiber2G = data2[:,2]
        Data_Fiber2R = data3[:,2]

    Data_1 = [Data_Fiber1iso, Data_Fiber1G, Data_Fiber1R]
    Data_2 = [Data_Fiber2iso, Data_Fiber2G, Data_Fiber2R]

    # convert to time in seconds
    time_seconds = np.arange(len(Data_Fiber1iso)) /sampling_rate 

    return Data_1, Data_2, PMts


# SAVE DATA: define the folder to save results
def define_resultsDir(folder):
    save_dir = r"/results"
    results_folder = os.path.join(save_dir,folder)

    # create the directory if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    print("results folder",results_folder)

    return results_folder