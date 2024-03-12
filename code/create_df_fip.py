
#%%
import os
import pandas as pd
import data_acquisition_library as dl
import numpy as np
import shutil
!pip install pathlib2
from pathlib2 import Path
import glob
from download_data import download_assets, search_assets, delete_downloaded_assets, get_data_ids

#%%
def clean_timestamps(timestamps, pace=0.05, tolerance=0.2):
    threshold_remove = pace - pace * tolerance
    threshold_fillin = pace + pace * tolerance
    idxs_to_remove = np.diff(timestamps, prepend=-np.Inf)<threshold_remove
    timestamps_cleaned = timestamps[~idxs_to_remove]
    timestamps_final = timestamps_cleaned
    while any(np.diff(timestamps_final)>threshold_fillin):
        idx_to_fillin = np.where(np.diff(timestamps_final)>threshold_fillin)[0][0]
        gap_to_fillin = np.round(np.diff(timestamps_final)[idx_to_fillin],2)    
        values_to_fillin = timestamps_final[idx_to_fillin]+np.arange(pace,gap_to_fillin,pace)
        timestamps_final = np.insert(timestamps_final, idx_to_fillin+1, values_to_fillin)
    return timestamps_final


#%% Import all datasets
data_assets = search_assets('FIP*')
subject_ids = [get_data_ids(data_asset)[0] for data_asset in data_assets]
subject_ids = np.unique(subject_ids)

#%%
N_subjects = len(subject_ids)
N_processes = N_subjects

for i_process in np.arange(N_processes):
    subject_id = subject_ids[i_process]
    folder = '../scratch/'+subject_id+'/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv', download_folder=folder)
    
    sessions = glob.glob(folder+'*')    
    for session in sessions:
        AnalDir = session
        Data_1, Data_2, PMts = dl.acquire_neural_data(AnalDir)   
        timestamps_fip = PMts
        mode_fip = statistics.mode(np.round(np.diff(timestamps_fip),2))
        if mode_fip != 0.05:
            print("wrong frequency")
            continue
        timestamps_fip_cleaned = clean_timestamps(timestamps_fip, pace=mode_fip)
        df_iter_cleaned = pd.merge(pd.DataFrame(timestamps_fip_cleaned, columns=['time']), df_iter, on='time', how='left')
        df_iter_cleaned.to_pickle('../results/'+AnalDir.strip('../scratch/')+os.sep+'df_fip.pkl')
        