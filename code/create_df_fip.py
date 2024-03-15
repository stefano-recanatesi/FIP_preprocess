
#%%
import os
import pandas as pd
import numpy as np
import shutil
!pip install boto3
!pip install pathlib2
!pip install aind_codeocean_api
import itertools
from pathlib2 import Path
import glob
from util import *

#%% Import all datasets
data_assets = search_all_assets('FIP*')
subject_ids = [get_data_ids(data_asset)[0] for data_asset in data_assets]
subject_ids = np.unique(subject_ids)

# REMOVE LATER
# subject_ids = ['FIP_642858_2022-10-31_14-41-15']

#%%
N_subjects = len(subject_ids)
N_processes = N_subjects
N_assets_per_subject = 2

df_fip = pd.DataFrame()
df_data_acquisition = pd.DataFrame()
df_pp_params = pd.DataFrame()
for i_process in np.arange(N_processes)[25:27]:
    subject_id = subject_ids[i_process]
    print(subject_id)
    # subject_id = '666610'
    folder = '../scratch/'+subject_id+'/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv', download_folder=folder, max_assets_to_download=N_assets_per_subject)
    
    sessions = glob.glob(folder+'*')    
    for session in sessions:
        print(subject_id + ' ' + session)
        AnalDir = session
        subject_id, session_date, session_time = (AnalDir.split('/')[-1]).strip('FIP_').split('_')
        df_fip_ses, df_data_acquisition_ses = data_to_dataframe(AnalDir)    
        df_fip_ses_cleaned = clean_timestamps_df(df_fip_ses)
        df_fip_ses_pp, df_pp_params_ses = preprocess_fip(df_fip_ses_cleaned, methods=['poly', 'exp'])
        df_fip_ses = pd.concat([df_fip_ses, df_fip_ses_pp])
        df_data_acquisition = pd.concat([df_data_acquisition, df_data_acquisition_ses])
        df_pp_params = pd.concat([df_pp_params, df_pp_params_ses])
        
        save_filename = '../results/'+AnalDir.strip('../scratch/')+os.sep+'df_fip.pkl'
        save_dirname = os.path.dirname(save_filename)
        if not os.path.exists(save_dirname):
            Path(save_dirname).mkdir(parents=True, exist_ok=True)            
        df_fip_ses.to_pickle(save_filename)
        df_pp_params_ses.to_pickle(save_filename[:-4]+'_pp.pkl')


#%%
# df_data_qc = df_data_acquisition.groupby(['ses_idx','system']).agg(np.nansum).reset_index()
# df_data_qc['subject_id'] = df_data_qc['ses_idx'].apply(lambda x: x[:6])
# df_data_qc = df_data_qc.groupby(['subject_id','ses_idx','system']).agg(np.nanmean).drop(columns='N_files')
# df_data_qc['sum'] = np.nansum(df_data_qc.values, axis=1)
# df_data_qc
# pd.value_counts(df_data_qc['sum'])
# df_data_qc = df_data_qc[df_data_qc['sum']!=6]
# # df_data_qc.groupby('subject_id')['sum'].agg(np.nanmean)
