
#%%
import os
import pandas as pd
import numpy as np
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
N_assets_per_subject = 2
N_subjects_to_process = 2

N_missing_values = []

df_fip = pd.DataFrame()
df_data_acquisition = pd.DataFrame()
df_pp_params = pd.DataFrame()
for i_process in np.arange(N_processes)[:N_subjects_to_process]:
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
        df_fip_ses_cleaned = clean_timestamps_df(df_fip_ses)
#        N_missing_values.append(pd.isna(df_fip_ses_cleaned.signal).sum())

#%%
        df_fip_ses_aligned, behavior_system = align_timestamps(df_fip_ses_cleaned, AnalDir) # Not using NWB anymore
        df_data_acquisition = pd.concat([df_data_acquisition, df_data_acquisition_ses])        
        
        #% Preprocessing of FIP data
        df_fip_ses_aligned.loc[:,'pp_method'] = 'None'        
        df_fip_ses_pp, df_pp_params_ses = preprocess_fip(df_fip_ses_aligned, methods=['poly', 'exp'])
        df_fip_ses = pd.concat([df_fip_ses_aligned, df_fip_ses_pp])        
        df_pp_params = pd.concat([df_pp_params, df_pp_params_ses])

        #% Storing of dataframes
        save_dirname = '../results/'+AnalDir.strip('../scratch/')
        save_filename_df_fip = save_dirname+os.sep+ ses_idx +'_df_fip.pkl'        
        if not os.path.exists(save_dirname):
            Path(save_dirname).mkdir(parents=True, exist_ok=True)            
        df_fip_ses.to_pickle(save_filename_df_fip)
        df_pp_params_ses.to_pickle(save_filename_df_fip[:-4]+'_pp.pkl')    

#%%