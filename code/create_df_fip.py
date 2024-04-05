
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
!pip install pynwb
!pip install boto3
!pip install pathlib2
!pip install aind_codeocean_api
import itertools
from pynwb import NWBFile, NWBHDF5IO
from pynwb.core import DynamicTable
from datetime import datetime
from pathlib2 import Path
import glob
from util import *
import re

#%% Import all datasets
data_assets = search_all_assets('FIP*')
subject_ids = [get_data_ids(data_asset)[0] for data_asset in data_assets]
subject_ids = np.unique(subject_ids)
# subject_ids = ['632104']
#%%
N_subjects = len(subject_ids)
N_processes = N_subjects
N_assets_per_subject = 200
N_subjects_to_process = 200

# lists to hold missing ts data

# number of missing timestamps across all sessions in all subjects
percentage_missing_values = []

#
All_delta_ts = []


# df_fip = pd.DataFrame() -- remove this yeah?

df_data_acquisition = pd.DataFrame()
df_pp_params = pd.DataFrame()
for i_process in np.arange(N_processes)[:N_processes]:
    subject_id = subject_ids[i_process]
    #subject_id='632106'
    print(subject_id)
    # folder = '/root/capsule/data/PhotometryForaging_AnalDev/'
    folder = '../scratch/'+subject_id+'/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv', download_folder=folder, max_assets_to_download=N_assets_per_subject)
#    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='TTL_', download_folder=folder, max_assets_to_download=N_assets_per_subject)
#    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='*', download_folder=folder, max_assets_to_download=N_assets_per_subject)

    sessions = glob.glob(folder+'*')    
    for session in sessions:
        print(subject_id + ' ' + session)
        AnalDir = session
        subject_id, session_date, session_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", AnalDir).group().split('_')            
        ses_idx = subject_id+'_'+session_date+'_'+session_time

        #% Creation of FIP dataframe
        df_fip_ses, df_data_acquisition_ses = data_to_dataframe(AnalDir) 

        # check if there was actually data - NEW
        if df_fip_ses.empty :
            print("mpty session")
            continue
        df_fip_ses_cleaned = clean_timestamps_df(df_fip_ses)


        # calculating the percentage of missing timestamps in the session
        n_missing_timestamps = pd.isna(df_fip_ses_cleaned.signal).sum()
        total_n_timestamps = df_fip_ses_cleaned.signal.shape[0]
        percentage_missing_values.append((n_missing_timestamps/total_n_timestamps)*100)
    
        # check how big the gap between consecutive timestamps is
        delta_ts = np.round(np.diff(df_fip_ses.time),3)
        All_delta_ts.append(delta_ts) # -- confirm this works the way I think


results_folder = r'/results'
os.makedirs(results_folder, exist_ok=True)

# PLOT 1   
# percentage of missing timestamps across all sessions in all subjects
unique_percents, n_sessions = np.unique(percentage_missing_values, return_counts=True)

# percentage of sessions with missing x timestamps 
# where x are unique values in percent_missing
fig1 = plt.figure()
percent_sessions = (n_sessions/(np.sum(n_sessions)))*100
plt.plot(unique_percents,percent_sessions,'o')
plt.xlabel('Percentage of missing timesteps')
plt.ylabel('Percentage of sessions')
plt.savefig(results_folder + os.sep +'plot1.png')

# PLOT 2

# Size of all intervals across all sessions and subjects
All_delta_ts = np.concatenate(All_delta_ts)

# make a histogram of these intervals. Expected is a gaussian around 0.05 (restrict to range because of 
# transitions between channels)
fig2 = plt.figure()
n_bins = int(np.sqrt(np.size(All_delta_ts)))

n, bins, patches = plt.hist(All_delta_ts, bins=n_bins,range=(-0.5,0.5), facecolor='green', alpha=0.75)
plt.xlabel('Interval between timestamps (ms)')
plt.ylabel('counts/bin')
plt.savefig(results_folder + os.sep +'plot2.png')

# alternative plot2
unique_deltas, counts = np.unique(All_delta_ts[All_delta_ts<1], return_counts=True)

percent_counts = (counts/(np.sum(counts)))*100

fig3 = plt.figure()
plt.plot(unique_deltas, percent_counts,'o', markersize=1)
plt.xlabel('Interval between timestamps (ms)')
plt.ylabel('Percentage of all timestamps')
plt.savefig(results_folder + os.sep +'plot3.png')

#%%
        # df_fip_ses_aligned, behavior_system = align_timestamps(df_fip_ses_cleaned, AnalDir) # Not using NWB anymore
        # df_data_acquisition = pd.concat([df_data_acquisition, df_data_acquisition_ses])        
        
        # #% Preprocessing of FIP data
        # df_fip_ses_aligned.loc[:,'pp_method'] = 'None'        
        # df_fip_ses_pp, df_pp_params_ses = preprocess_fip(df_fip_ses_aligned, methods=['poly', 'exp'])
        # df_fip_ses = pd.concat([df_fip_ses_aligned, df_fip_ses_pp])        
        # df_pp_params = pd.concat([df_pp_params, df_pp_params_ses])

        # #% Storing of dataframes

        
        # save_dirname = '../results/'
        
        # save_filename_df_fip = save_dirname+os.sep+ ses_idx +'_df_fip.pkl'        
        # if not os.path.exists(save_dirname):
        #     Path(save_dirname).mkdir(parents=True, exist_ok=True)            
        # df_fip_ses.to_pickle(save_filename_df_fip)
        # df_pp_params_ses.to_pickle(save_filename_df_fip[:-4]+'_pp.pkl')    

#%%