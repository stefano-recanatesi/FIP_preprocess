#%%
import os
import boto3
import aind_codeocean_api
import json
import boto3
import numpy as np
from aind_codeocean_api.codeocean import CodeOceanClient
from botocore.exceptions import ClientError
from pynwb import NWBHDF5IO
import pandas as pd
import shutil
import itertools
import Preprocessing_library as pp
from pathlib2 import Path
import statistics
import glob
import re

# datafolder =  '../results/'
CO_TOKEN = 'cop_YTg1NWZkMGE1ZTc3NDAwZGJiNDU5NTViMDE0Y2UzNDZHZmlKTFFwVWlQb3psWG10NUx3ZE9jcXVDTkVXSHZWSjIyNzhkZTZj'
CO_DOMAIN = "https://codeocean.allenneuraldynamics.org"

def search_assets(query_asset='FIP_*'):    
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    response = co_client.search_all_data_assets(query="name:"+query_asset)
    data_assets_all = response.json()["results"]    
    data_assets = [r for r in data_assets_all if query_asset.strip('*') in r["name"]]
    return data_assets

def search_all_assets(query_asset='FIP_*', **kwargs):
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    has_more = True
    start = 0
    limit = 50    
    data_assets = []
    while has_more:
        response = co_client.search_data_assets(start=start, limit=limit, query='name:'+query_asset, **kwargs).json()
        has_more = response['has_more']
        results = response['results']
        data_assets_temp = [r for r in results if query_asset.strip('*') in r["name"]]
        data_assets = data_assets + data_assets_temp
        start += len(results)
    return data_assets

def download_assets(query_asset='FIP_*', query_asset_files='.csv',download_folder='../scratch/', max_assets_to_download=3):
    co_client = CodeOceanClient(domain=CO_DOMAIN, token=CO_TOKEN)
    response = co_client.search_all_data_assets(query="name:"+query_asset)
    data_assets_all = response.json()["results"]
    # Filter if data in asset name
    data_assets = [r for r in data_assets_all if query_asset.strip('*') in r["name"]]

    # Create s3 client
    s3_client = boto3.client('s3')
    s3_response = s3_client.list_buckets()
    s3_buckets = s3_response["Buckets"]

    for asset in data_assets[:max_assets_to_download]:
        # Get bucket id for datasets    
        dataset_bucket_prefix = asset['sourceBucket']['bucket']
        asset_bucket = [r["Name"] for r in s3_buckets if dataset_bucket_prefix in r["Name"]][0]

        asset_name = asset["name"]
        asset_id = asset["id"]
        matching_string = query_asset_files

        response = s3_client.list_objects_v2(Bucket=asset_bucket, Prefix=asset_name)
        for object in response['Contents']:
            if (asset_name in object['Key']) and (matching_string in object['Key']):            
                filename = os.path.join(download_folder, object['Key'])
                pathname = os.path.dirname(filename)
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                # print('Downloading ' + filename)
                s3_client.download_file(asset_bucket, object['Key'], filename)

    s3_client.close()

def delete_downloaded_assets(query_asset='FIP_', query_asset_files='*.csv',download_folder='../scratch/', delete_folders=True):
    folders = [x[0] for x in os.walk(download_folder) if query_asset in x[0]]
    folders = np.sort(folders)[::-1]
    for folder in folders:
        if delete_folders:
            shutil.rmtree(folder)
        else:
            filenames = glob.glob(os.path.join(folder,query_asset_files), recursive=True)
            for filename in filenames:
                os.remove(filename)

def get_data_ids(data_asset):
    data_ids = data_asset['name'].strip('FIP_').split('_')
    return data_ids

#%%

# function to clean the timestamps
def clean_timestamps(timestamps, pace=0.05, tolerance=0.2):
    
    # remove timestamps that are too close to each other
    threshold_remove = pace - pace * tolerance
    idxs_to_remove = np.diff(timestamps, prepend=-np.Inf)<threshold_remove
    timestamps_cleaned = timestamps[~idxs_to_remove]

    # timestamps that are too far apart
    threshold_fillin = pace + pace * tolerance
    timestamps_final = timestamps_cleaned

    # loops over all identified gaps
    while any(np.diff(timestamps_final)>threshold_fillin):
        # the first index in the diff array
        idx_to_fillin = np.where(np.diff(timestamps_final)>threshold_fillin)[0][0]

        # round the values in the diff array to 2 dp
        gap_to_fillin = np.round(np.diff(timestamps_final)[idx_to_fillin],2)

        # gets values of timestamps to add to the gap -- proportional to the size of the gap
        values_to_fillin = timestamps_final[idx_to_fillin]+np.arange(pace,gap_to_fillin,pace)

        # then inserts these values to the final timestamps array
        timestamps_final = np.insert(timestamps_final, idx_to_fillin+1, values_to_fillin)

    return timestamps_final
 

def load_NPM_fip_data(filenames, channels_per_file=2):
    df_fip = pd.DataFrame()
    df_data_acquisition = pd.DataFrame()
    for filename in filenames:
        subject_id, session_date, session_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename).group().split('_')
        # subject_id, session_date, session_time = (os.path.dirname(filename).split('/')[-1]).strip('FIP_').split('_')
        ses_idx = subject_id+'_'+session_date+'_'+session_time
        df_fip_file = pd.read_csv(filename)                       
        name = os.path.basename(filename)[:4]
        channel = {'L415':'G', 'L470':'G', 'L560':'R'}[name]
        columns = [col for col in df_fip_file.columns if ('Region' in col) & (channel==col[-1])]
        columns = np.sort(columns)[:channels_per_file]
        df_file = pd.DataFrame()
        for i_col, col in enumerate(columns):
            channel = {'L415':'Iso', 'L470':'G', 'L560':'R'}[name]
            channel_number = i_col                    
            df_fip_file_renamed = df_fip_file.loc[:,['FrameCounter', 'Timestamp', col]]                    
            df_fip_file_renamed = df_fip_file_renamed.rename(columns={'FrameCounter':'frame_number', 'Timestamp':'time', col:'signal'})                    
            df_fip_file_renamed.loc[:, 'channel'] = channel
            df_fip_file_renamed.loc[:, 'channel_number'] = channel_number                                                
            df_file = pd.concat([df_file, df_fip_file_renamed])                        
            df_data_acquisition = pd.concat([df_data_acquisition, pd.DataFrame({'ses_idx':ses_idx, 'system':'NPM', channel+str(channel_number):1.,'N_files':len(filenames)}, index=[0])])                                                       
        df_fip = pd.concat([df_fip, df_file])   
    df_fip.loc[:,'system'] = 'NPM'        
    df_fip['ses_idx'] = subject_id+'_'+session_date+'_'+session_time
    df_fip = df_fip[['ses_idx', 'system', 'channel', 'channel_number', 'time', 'signal']]
    return df_fip, df_data_acquisition


def load_Homebrew_fip_data(filenames, channels_per_file=2):                        
    df_fip = pd.DataFrame()
    df_data_acquisition = pd.DataFrame()
    save_fip_channels= np.arange(1, channels_per_file+1)
    for filename in filenames:
        subject_id, session_date, session_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename).group().split('_')
        #  = (os.path.dirname(filename).split('/')[-1]).strip('FIP_').split('_')
        ses_idx = subject_id+'_'+session_date+'_'+session_time       
        header = os.path.basename(filename).split('/')[-1]
        channel = '_'.join(header.split('_')[:2]).strip('FIP_Data')        
        try:
            df_fip_file = pd.read_csv(filename, header=None)  #read the CSV file        
        except pd.errors.EmptyDataError:
            continue
        except FileNotFoundError:
            continue
        df_file = pd.DataFrame()
        for col in df_fip_file.columns[save_fip_channels]:
            df_fip_file_renamed = df_fip_file[[0, col]].rename(columns={0:'time', col:'signal'})
            channel_number = int(col)
            df_fip_file_renamed['channel_number'] = channel_number
            df_fip_file_renamed.loc[:, 'frame_number'] = df_fip_file.index.values
            df_file = pd.concat([df_file, df_fip_file_renamed])
            df_data_acquisition = pd.concat([df_data_acquisition, pd.DataFrame({'ses_idx':ses_idx, 'system':'FIP', channel+str(channel_number):1.,'N_files':len(filenames)}, index=[0])])                                           
        df_file['channel'] = channel            
        df_fip = pd.concat([df_fip, df_file], axis=0)     
    if len(df_fip) > 0:       
        df_fip['system'] = 'FIP' 
        df_fip['ses_idx'] = subject_id+'_'+session_date+'_'+session_time
        df_fip = df_fip[['ses_idx', 'system', 'channel', 'channel_number', 'time', 'signal']]
    return df_fip, df_data_acquisition


def data_to_dataframe(AnalDir, channels_per_file=2):    
    print('preprocessing Neurophotometrics Data')
    #% Processing NPM system
    filenames = []    
    for name in ['L415', 'L470', 'L560']:
        if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
            filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True))
            system = 'NPM'
   
    for name in ['FIP_DataG', 'FIP_DataR', 'FIP_DataIso']:
        if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
            filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True)) 
            system = 'Homebrew'
    
    # NEW: statement to catch the sessions with no data: output empty df
    if 'system' in locals():

        if system == 'NPM':
            df_fip, df_data_acquisition = load_NPM_fip_data(filenames, channels_per_file)
        if system == 'Homebrew':
            df_fip, df_data_acquisition = load_Homebrew_fip_data(filenames, channels_per_file)

    else:
        print('No data found')
        df_fip = pd.DataFrame()
        df_data_acquisition = pd.DataFrame()

    return df_fip, df_data_acquisition


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


def clean_timestamps_df(df_fip_ses):
    if not len(df_fip_ses):
        return df_fip_ses    
    channels = pd.unique(df_fip_ses['channel']) # ['G', 'R', 'Iso']        
    df_fip = pd.DataFrame()
    for i_channel, channel in enumerate(channels):
        df_iter_channel = df_fip_ses[df_fip_ses['channel']==channel]        
        channel_numbers = pd.unique(df_iter_channel['channel_number'])
        for channel_number in channel_numbers:
            df_iter = df_iter_channel[df_iter_channel['channel_number']==channel_number]        
            
            if not len(df_iter):
                continue                        
            if np.max(np.diff(df_iter['time'])) > 10:
                df_iter.loc[:,'time'] = df_iter['time'] / 1000.  #convert to seconds- signal milliseconds
            timestamps_fip = df_iter['time'].values

            mode_fip = statistics.mode(np.round(np.diff(timestamps_fip),2))
            frequency = int(1/mode_fip)
            if (frequency!=20):
                print('    ' + channel+' The frenquencies of the timesteps are FIP '+str(frequency))            

            timestamps_fip_cleaned = clean_timestamps(timestamps_fip, pace=mode_fip)            
            if len(timestamps_fip_cleaned) != len(timestamps_fip):
                print('The timestamps were modified with a new length of '+ str(len(timestamps_fip_cleaned)) + ' from ' + str(len(timestamps_fip)))        

            # timestamps_fip_aligned =  all(np.unique(np.round(2*np.diff(timestamps_fip_cleaned),1))==0.1)                        
            df_iter_cleaned = pd.merge(pd.DataFrame(timestamps_fip_cleaned, columns=['time']), df_iter, on='time', how='left')            
            
            df_fip = pd.concat([df_fip, df_iter_cleaned], axis=0)      
    return df_fip


def clean_timestamps_Harp(timestamps_Harp):
    # io = NWBHDF5IO(nwb_filename, mode='r')
    # nwb = io.read()
    # subject_id, session_date, session_json_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", nwb_filename).group().split('_')
        
    # key_from_acq = ['FIP_falling_time', 'FIP_rising_time']    
    # events_ses = {key: nwb.acquisition[key].timestamps[:] for key in key_from_acq}    
    # timestamps_Harp_rising = events_ses['FIP_rising_time']
    # timestamps_Harp_falling = events_ses['FIP_falling_time']
    # timestamps_Harp = timestamps_Harp_rising    
    mode_Harp = statistics.mode(np.round(np.diff(timestamps_Harp),2))
    timestamps_Harp_cleaned = clean_timestamps(timestamps_Harp, pace=mode_Harp)
    timestamps_Harp_aligned =  all(np.unique(np.round(2*np.diff(timestamps_Harp_cleaned),1))==0.1)
    if not timestamps_Harp_aligned:
        print('Harp timestamps not succesfully cleaned')
        return np.nan
    return timestamps_Harp_cleaned

         
    

# for filename in filenames_fip[:]:
def preprocess_fip(df_fip, methods=['poly', 'exp']):
    df_fip_pp = pd.DataFrame()    
    df_pp_params = pd.DataFrame() 
    
    # df_fip = pd.read_pickle(filename)        
    if len(df_fip) == 0:
        return df_fip, df_pp_params

    sessions = pd.unique(df_fip['ses_idx'].values)
    sessions = sessions[~pd.isna(sessions)]
    channel_numbers = np.unique(df_fip['channel_number'].values)    
    channels = pd.unique(df_fip['channel']) # ['G', 'R', 'Iso']    
    channels = channels[~pd.isna(channels)]
    for pp_name in methods:     
        if pp_name in ['poly', 'exp']:   
            for i_iter, (channel, channel_number, ses_idx) in enumerate(itertools.product(channels, channel_numbers, sessions)):            
                df_fip_iter = df_fip[(df_fip['ses_idx']==ses_idx) & (df_fip['channel_number']==channel_number) & (df_fip['channel']==channel)]        
                if len(df_fip_iter) == 0:
                    continue
                
                NM_values = df_fip_iter['signal'].values   
                try:      
                    NM_preprocessed, NM_fitting_params = pp.tc_preprocess(NM_values, method=pp_name)
                except:
                    continue                                       
                df_fip_iter['signal'] = NM_preprocessed                            
                df_fip_iter['pp_method'] = pp_name
                df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], axis=0)                    
                
                NM_fitting_params.update({'pp_method':pp_name, 'channel':channel, 'channel_number':channel_number, 'ses_idx':ses_idx})
                df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)        
        if pp_name in ['double_exp']:
            for i_iter, (channel, channel_number, ses_idx) in enumerate(itertools.product(channels, channel_numbers, sessions)):            
                df_fip_iter = df_fip[(df_fip['ses_idx']==ses_idx) & (df_fip['channel_number']==channel_number)]        
                F = list()
                for i_channel, channel in enumerate(['iso', 'G', 'R']):
                    F.append(df_fip_iter[df_fip_iter['channel'] == channel].signal.values.flatten())
                F = np.vstack(F)
                dff_mc = preprocess(F)
                for i_channel, channel in enumerate(['iso', 'G', 'R']):
                    df_fip_channel = df_fip_iter[df_fip_iter['channel'] == channel]
                    df_fip_channel.loc[:,'signal'] = dff_mc[i_channel]
                    df_fip_channel['pp_method'] = pp_name
                df_fip_pp = pd.concat([df_fip_pp, df_fip_channel], axis=0)                    
                
                NM_fitting_params.update({'pp_method':pp_name, 'channel':channel, 'channel_number':channel_number, 'ses_idx':ses_idx})
                df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)        

    return df_fip_pp, df_pp_params
    

def align_timestamps(df_fip_ses_cleaned, AnalDir):
    file_TTL = glob.glob(AnalDir + os.sep + '**'+ os.sep + "TTL*")
    if len(file_TTL):
        df_fip_ses_aligned = alignment_fip_time_to_TTL(df_fip_ses_cleaned, AnalDir)
        behavior_system = 'TTL'

    file_harp = glob.glob(AnalDir+ '**'+ os.sep + 'TrainingFolder/*.json')        
    if len(file_harp):
        jsonFile = open(file_harp[0], 'r')
        values = json.load(jsonFile)
        key_rising_time = [key for key in values.keys() if 'Rising' in key]
        timestamps_Harp = values[key_rising_time]
        timestamps_Harp_cleaned = clean_timestamps_Harp(timestamps_Harp)
        df_fip_ses_aligned = alignment_fip_time_to_harp(df_fip_ses_cleaned, timestamps_Harp_cleaned)
        behavior_system = 'Bonsai'
    return df_fip_ses_aligned, behavior_system


# def alignment_fip_time_to_TTL(df_fip_ses, AnalDir):       
#     filename_ts = glob.glob(AnalDir + os.sep + '**' + os.sep + "TimeStamp_*")
#     if len(filename_ts):  # data from NPM
#         ts_10kHz = pd.read_csv(filename_ts[0], header=None).values/100000        
#         pace_10kHz_signal = statistics.mode(np.round(np.diff(ts),2))
#         if pace_10kHz_signal != 0.05:
#             pace_10kHz_signal = np.nan * pace_10kHz_signal
#             print("The frequency of the signal is not 20Hz")
#     else:
#         ts_10kHz = np.nan * np.ones(df_fip_ses.shape[0])

#     df_fip_ses_aligned = pd.DataFrame()
#     cols_conditions = np.array([col for col in df_fip_ses.columns if col not in ['signal', 'time']])
#     df_conditions = df_fip_ses[cols_conditions].drop_duplicates().reset_index(drop=True)    
#     for i_condition in range(df_conditions.shape[0]):
#         condition_vals = df_conditions.iloc[[i_condition],:]        
#         condition_bools = (df_fip_ses[cols_conditions]==condition_vals.values).apply(all, axis=1)
#         df_iter = df_fip_ses.loc[condition_bools, :]
#         channel = condition_vals['channel'].values[0]
#         if pd.isna(channel):
#             continue        
#         timestampts_fip = df_iter['time'].values
#         df_iter.loc[:,'time_fip'] = np.round(timestampts_fip,4)
        
#         ts_10kHz = ts_10kHz[np.linspace(0,ts_10kHz.shape[0]-1,len(timestamps_fip)).astype(int),:]  # length of this must be the same as that of GCaMP_dF_F
#         ts = ts_10kHz[:,0]

#         df_iter.loc[:, 'time'] = ts
#         df_fip_ses_aligned = pd.concat([df_fip_ses_aligned, df_iter])

#     return df_fip_ses_aligned

def align_fip_time_to_harp(df_fip_ses, timestamps_Harp_cleaned):
    if not len(df_fip_ses):
        return df_fip_ses    
    
    channel_number = pd.unique(df_fip_ses['channel_number'])[0]
    channels = pd.unique(df_fip_ses['channel'])    
    channels = channels[~pd.isna(channels)]
    df_fip_sel = df_fip_ses[(df_fip_ses['pp_method']=='None') & (df_fip_ses['channel_number']==channel_number)]

    timestamps = {}
    for channel in channels:
        timestamps[channel] = df_fip_sel[df_fip_sel['channel'] == channel]['time'].values
    
    if len(np.unique([len(timestamps[channel]) for channel in channels])) == 1: # This means that all channels have the same amount of timestamps
        reference_channel = 'G'
    else:
        idx = np.argmax([len(timestamps[key]) for key in timestamps.keys()])
        reference_channel = list(timestamps.keys())[idx]
        subsample = 'True'

    timestamps_ref = timestamps[reference_channel]
    timestamps_Harp_padded = np.insert(timestamps_Harp_cleaned, 0, [np.nan]*(len(timestamps_ref)-len(timestamps_Harp_cleaned)))
    
    timestamps_new = {}
    for channel in channels:            
        sub_idxs = np.linspace(0, len(timestamps_Harp_padded)-1, len(timestamps[channel])).astype(int)
        time_shifts = timestamps[channel] - timestamps_ref[sub_idxs]        
        timestamps_new[channel] = timestamps_Harp_padded[sub_idxs] + time_shifts
    
    df_fip_ses_aligned = pd.DataFrame()
    cols_conditions = np.array([col for col in df_fip_ses.columns if col not in ['signal', 'time']])
    df_conditions = df_fip_ses[cols_conditions].drop_duplicates().reset_index(drop=True)    
    for i_condition in range(df_conditions.shape[0]):
        condition_vals = df_conditions.iloc[[i_condition],:]        
        condition_bools = (df_fip_ses[cols_conditions]==condition_vals.values).apply(all, axis=1)
        df_iter = df_fip_ses.loc[condition_bools, :]
        channel = condition_vals['channel'].values[0]
        if pd.isna(channel):
            continue        
        df_iter.loc[:,'time_fip'] = np.round(df_iter['time'].values,4)
        df_iter.loc[:, 'time'] = timestamps_new[channel]
        df_fip_ses_aligned = pd.concat([df_fip_ses_aligned, df_iter])
    
    return df_fip_ses_aligned

    
def compute_timestamps_bitcodes(AnalDir):
    TTL_signal = np.fromfile(glob.glob(AnalDir + os.sep + '**'+  os.sep + "TTL_20*")[0])
    file_TTLTS = glob.glob(AnalDir + os.sep + '**'+ os.sep + "TTL_TS*")[0]
    TTL_ts = pd.read_csv(file_TTLTS, header=None).values.flatten()
            
    #% Sorting NIDAQ-AI channels
    if (len(TTLsignal)/1000) / len(TTLts) == 1:
        print("Num Analog Channel: 1")
        
    elif (len(TTLsignal)/1000) / len(TTLts) == 2:  #this shouldn't happen, though...
        TTL_signal_ini = TTL_signal[0::2]
        print("Num Analog Channel: 2")
            
    elif (len(TTLsignal)/1000) / len(TTLts) >= 3:
        TTL_signal_ini = TTL_signal[0::3]    
        print("Num Analog Channel: 3")
    else:
        print("Something is wrong with TimeStamps or Analog Recording...")
        
    #% analoginputs binalize    
    TTL_signal = (TTL_signal_ini > 3).astype(int)
    diff = np.diff(TTL_signal, prepend=0)
    
    # Find indices where diff is 1
    indices = np.where(diff == 1)[0]

    # For each index, find the next index where diff is -1 within a range of 120
    next_indices = [np.where(diff[ii:ii+120] == -1)[0][0] for ii in indices]

    # Convert lists to numpy arrays
    TTL_p = np.array(indices)
    TTL_l = np.array(next_indices)

    #This cleans up the TTL signal into predetermined values
    values = np.array([ 1.,  2.,  3., 10., 20., 30., 40.])
    bin_edges = np.concatenate([values[:-1]+np.diff(values)/2, [values[-1]+10]])
    idxs = np.digitize(TTL_l, bin_edges)
    TTL_l = values[idxs]

    #%
    # Calculate ind_tmp and dec_tmp for all elements in TTL_p
    ind_tmp = np.ceil(TTL_p / 1000).astype(int) - 2
    dec_tmp = TTL_p / 1000 + 1 - np.ceil(TTL_p / 1000)

    # Filter out indices where ind_tmp is greater than or equal to the length of TTLts
    valid_indices = ind_tmp < len(TTL_ts)
    ind_tmp = ind_tmp[valid_indices]
    dec_tmp = dec_tmp[valid_indices]

    # Calculate ms_target for all valid indices
    ms_target = TTL_ts[ind_tmp]

    # Calculate idx for all valid indices
    idx = np.argmin(np.abs(ts_10kHz[:, 0, None] - ms_target - dec_tmp * 1000), axis=0)

    # Calculate residual for all valid indices
    residual = ts_10kHz[idx, 0] - ms_target - dec_tmp * 1000

    # Calculate TTL_p_align and TTL_p_align_1k
    TTL_p_align = idx
    TTL_p_align_1k = time_seconds[idx] - residual / 1000

    TTL_l_align = TTL_l[0:len(TTL_p_align)]

    # Filter indices where TTL_l is 20
    indices = np.where(TTL_l == 20)[0]

    # Extract relevant data
    BarcodeP = TTL_p[indices]
    BarcodeP_1k = TTL_p_align_1k[indices]

    # Calculate the indices for TTLsignal1
    offsets = np.arange(20) * 20 + 30 + 5
    indices = BarcodeP[:, np.newaxis] + offsets

    # Vectorized operation to populate BarcodeBin
    BarcodeBin = TTL_signal_ini[indices]

    # Convert BarcodeBin to BarChar
    bitcodes = np.array([''.join(map(str, row.astype(int))) for row in BarcodeBin])
#add bitcodes storing
    return timestamps_bitcodes, bitcodes

def align_bitcodes_nwb_to_fip(filename_nwb, df_fip_ses_aligned, AnalDir):
    timestamps_bitcodes_behavior, bitcodes_behavior = load_behavior_bitcodes_timestamps(filename_nwb)
    timestamps_bitcodes_fip, bitcodes_fip = compute_timestamps_bitcodes(AnalDir)
    new_timestamps_fip = align_bitcodes(timestamps_bitcodes_behavior, bitcodes_behavior, timestamps_bitcodes_fip, bitcodes_fip)
    df_fip_ses_aligned_to_nwb = realign_timestamps(df_fip_ses_aligned, new_timestamps_fip)
    return df_fip_ses_aligned_to_nwb

def alignment_fip_time_to_TTL(df_fip_ses_cleaned, AnalDir):       
    filename_ts = glob.glob(AnalDir + os.sep + '**' + os.sep + "TimeStamp_*")
    if len(filename_ts):  # data from NPM
        ts_10kHz = pd.read_csv(filename_ts[0], header=None).values/100000    
        pace_10kHz_signal = statistics.mode(np.round(np.diff(ts_10kHz[:,0]),2))
        if pace_10kHz_signal != 0.05:
            pace_10kHz_signal = np.nan * pace_10kHz_signal
            print("The frequency of the signal is not 20Hz")
    else:
        ts_10kHz = np.nan * np.ones(df_fip_ses_cleaned.shape[0])

    df_fip_ses_aligned = pd.DataFrame()
    cols_conditions = np.array([col for col in df_fip_ses_cleaned.columns if col not in ['signal', 'time']])
    df_conditions = df_fip_ses_cleaned[cols_conditions].drop_duplicates().reset_index(drop=True)    
    for i_condition in range(df_conditions.shape[0]):
        condition_vals = df_conditions.iloc[[i_condition],:]        
        condition_bools = (df_fip_ses_cleaned[cols_conditions]==condition_vals.values).apply(all, axis=1)
        df_iter = df_fip_ses_cleaned.loc[condition_bools, :]
        channel = condition_vals['channel'].values[0]
        if pd.isna(channel):
            continue        
        timestamps_fip = df_iter['time'].values
        df_iter.loc[:,'time_fip'] = np.round(timestamps_fip,4)
        
        ts_10kHz = ts_10kHz[np.linspace(0,ts_10kHz.shape[0]-1,len(timestamps_fip)).astype(int),:]  # length of this must be the same as that of GCaMP_dF_F
        ts = ts_10kHz[:,0]

        df_iter.loc[:, 'time'] = ts
        df_fip_ses_aligned = pd.concat([df_fip_ses_aligned, df_iter])

    return df_fip_ses_aligned
# def function(timestamps_fip, TTL_p_align_1k, bitcodes, TTL_pulses_Han_time, bitcodes_Han)
