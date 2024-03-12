
#%%
import os
import pandas as pd
import data_acquisition_library as dl
import numpy as np
import shutil
!pip install boto3
!pip install pathlib2
!pip install aind_codeocean_api
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

def data_to_dataframe(AnalDir):    
    print('preprocessing Neurophotometrics Data')
    #% Processing NPM system
    filenames = []
    for name in ['L415', 'L470', 'L560']:
        if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
            filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True))

    if len(filenames):
        df_fip = pd.DataFrame()
        df_data_acquisition = pd.DataFrame()
        for filename in filenames:
            subject_id, session_date, session_time = (AnalDir.split('/')[-1]).strip('FIP_').split('_')
            df_fip_file = pd.read_csv(filename)                       
            name = os.path.basename(filename)[:4]
            channel = {'L415':'G', 'L470':'G', 'L560':'R'}[name]
            columns = [col for col in df_fip_file.columns if ('Region' in col) & (channel==col[-1])]
            columns = np.sort(columns)
            df_file = pd.DataFrame()
            for i_col, col in enumerate(columns):
                channel = {'L415':'Iso', 'L470':'G', 'L560':'R'}[name]
                channel_number = i_col                    
                df_fip_file_renamed = df_fip_file.loc[:,['FrameCounter', 'Timestamp', col]]                    
                df_fip_file_renamed = df_fip_file_renamed.rename(columns={'FrameCounter':'frame_number', 'Timestamp':'time', col:'signal'})                    
                df_fip_file_renamed.loc[:, 'channel'] = channel
                df_fip_file_renamed.loc[:, 'channel_number'] = channel_number
                df_fip_file_renamed.loc[:, 'system'] = 'NPM'
                ses_idx = subject_id+'_'+session_date+'_'+session_time
                df_fip_file_renamed.loc[:, 'ses_idx'] = ses_idx
                df_file = pd.concat([df_file, df_fip_file_renamed])                        
                df_data_acquisition = pd.concat([df_data_acquisition, pd.DataFrame({'ses_idx':ses_idx, 'system':'NPM', channel+str(channel_number):1.,}, index=[0])])                
            df_fip = pd.concat([df_fip, df_file])                        

    #% Processing Homebrew FIP system
    filenames = []
    save_fip_channels=[1,2]
    # AnalDir = '/root/capsule/data/PhotometryForaging_AnalDev/697062_2023-11-15_10-02-05'
    for name in ['FIP_DataG', 'FIP_DataR', 'FIP_DataIso']:
        if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
            filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True))                       
    if len(filenames):
        df_fip = pd.DataFrame()
        df_data_acquisition = pd.DataFrame()
        for filename in filenames:
            subject_id, session_date, session_time = (AnalDir.split('/')[-1]).strip('FIP_').split('_')
            filename_data = glob.glob(filename)[0]    
            header = filename_data.split('/')[-1]
            header = '_'.join(header.split('_')[:2])         
            try:
                df_fip_file = pd.read_csv(filename_data, header=None)  #read the CSV file        
            except pd.errors.EmptyDataError:
                continue
            except FileNotFoundError:
                continue
            df_file = pd.DataFrame()
            for col in df_fip_file.columns[save_fip_channels]:
                df_fip_file_renamed = df_fip_file[[0, col]].rename(columns={0:'time', col:'signal'})
                df_fip_file_renamed['channel_number'] = int(col)
                df_fip_file_renamed.loc[:, 'frame_number'] = df_fip_file.index.values
                df_file = pd.concat([df_file, df_fip_file_renamed])
            df_file['channel'] = header.replace('FIP_Data','')                
            df_fip = pd.concat([df_fip, df_file], axis=0)     
        df_fip['system'] = 'FIP' 
        df_fip['ses_idx'] = subject_id+'_'+session_date+'_'+session_time

    return df_fip, df_data_acquisition


#%% Import all datasets
data_assets = search_assets('FIP*')
subject_ids = [get_data_ids(data_asset)[0] for data_asset in data_assets]
subject_ids = np.unique(subject_ids)

#%%
N_subjects = len(subject_ids)
N_processes = N_subjects

df_fip = pd.DataFrame()
df_data_acquisition = pd.DataFrame()
for i_process in np.arange(N_processes)[:]:
    subject_id = subject_ids[i_process]
    # subject_id = '666610'
    folder = '../scratch/'+subject_id+'/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv', download_folder=folder, max_assets_to_download=2)
    
    sessions = glob.glob(folder+'*')    
    for session in sessions[:1]:
        AnalDir = session
        subject_id, session_date, session_time = (AnalDir.split('/')[-1]).strip('FIP_').split('_')
        df_fip_ses, df_data_acquisition_ses = data_to_dataframe(AnalDir)    
        df_fip = pd.concat([df_fip, df_fip_ses])
        df_data_acquisition = pd.concat([df_data_acquisition, df_data_acquisition_ses])

    delete_downloaded_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv',download_folder='../scratch/', delete_folders=True)
df_data_acquisition.groupby(['ses_idx','system']).agg(np.nansum)


#%%
        # Data_1, Data_2, PMts = dl.acquire_neural_data(AnalDir)   
        # break
        # timestamps_fip = PMts
        # mode_fip = statistics.mode(np.round(np.diff(timestamps_fip),2))
        # if mode_fip != 0.05:
        #     print("wrong frequency")
        #     continue
        # timestamps_fip_cleaned = clean_timestamps(timestamps_fip, pace=mode_fip)
        # df_iter_cleaned = pd.merge(pd.DataFrame(timestamps_fip_cleaned, columns=['time']), df_iter, on='time', how='left')
        # df_iter_cleaned.to_pickle('../results/'+AnalDir.strip('../scratch/')+os.sep+'df_fip.pkl')

    
        
#%%



#%%