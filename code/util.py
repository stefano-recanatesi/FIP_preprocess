#%%
import os
import boto3
import aind_codeocean_api
import boto3
import glob
import numpy as np
from aind_codeocean_api.codeocean import CodeOceanClient
from botocore.exceptions import ClientError
import pandas as pd
import shutil
import itertools
import Preprocessing_library as pp
from pathlib2 import Path
import statistics
import glob

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

# Update to download everything in the FIP subfolder
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

def load_NPM_fip_data(filenames, channels_per_file=2):
    df_fip = pd.DataFrame()
    df_data_acquisition = pd.DataFrame()
    for filename in filenames:
        subject_id, session_date, session_time = [i for i in os.path.dirname(filename).split('/') if i[:4]=='FIP_'][0].strip('FIP_').split('_')
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
        subject_id, session_date, session_time = [i for i in os.path.dirname(filename).split('/') if i[:4]=='FIP_'][0].strip('FIP_').split('_')
        #  = (os.path.dirname(filename).split('/')[-1]).strip('FIP_').split('_')
        ses_idx = subject_id+'_'+session_date+'_'+session_time
        filename_data = glob.glob(filename)[0]    
        header = filename_data.split('/')[-1]
        header = '_'.join(header.split('_')[:2])         
        channel = header.replace('FIP_Data','')
        try:
            df_fip_file = pd.read_csv(filename_data, header=None)  #read the CSV file        
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
    
    if system == 'NPM':
        df_fip, df_data_acquisition = load_NPM_fip_data(filenames, channels_per_file=2)
    if system == 'Homebrew':
        df_fip, df_data_acquisition = load_Homebrew_fip_data(filenames, channels_per_file=2)
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
        for i_iter, (channel, channel_number, ses_idx) in enumerate(itertools.product(channels, channel_numbers, sessions)):            
            df_fip_iter = df_fip[(df_fip['ses_idx']==ses_idx) & (df_fip['channel_number']==channel_number) & (df_fip['channel']==channel)]        
            if len(df_fip_iter) == 0:
                continue
            
            NM_values = df_fip_iter['signal'].values            
            NM_preprocessed, NM_fitting_params = pp.tc_preprocess(NM_values, method=pp_name)
            
            df_fip_iter['signal'] = NM_preprocessed                            
            df_fip_iter['pp_method'] = pp_name
            df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], axis=0)                    
            
            NM_fitting_params.update({'pp_method':pp_name, 'channel':channel, 'channel_number':channel_number, 'ses_idx':ses_idx})
            df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
            df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)        
    
    return df_fip_pp, df_pp_params

#%%

# Function to align the behaviour and NPM machines
# takes as input the length of the traces and the analysis directory
def alignment(len_traces, Analysis_dir):
    
    # get the file with the timestamps for TTL
    file_TTLTS = glob.glob(Analysis_dir + os.sep + "**" + os.sep + "TTL_TS*", recursive=True)[0]

    # get the TTL signal
    TTLsignal = np.fromfile(glob.glob(Analysis_dir + os.sep + "**" + os.sep + "TTL_20*", recursive=True)[0])
    
    # UPDATE: depending on the shape of Traces 
    time_seconds= np.arange(len_traces)/20

    # Get the Timestamps for the NPM machine -- UPDATE
    if glob.glob(Analysis_dir + os.sep + "**" + os.sep + "TimeStamp_*", recursive=True) != []:   #data from NPM
        file_TS = glob.glob(Analysis_dir + os.sep + "**" + os.sep + "TimeStamp_*", recursive=True)[0]
        
        with open(file_TS) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])

            # UPDATE: Get Cleaned timestamps
            PMts = datatemp[0:,:].astype(np.float32)
            # PMts = clean_timestamps
        
        
        PMts2 = PMts[0:len(PMts):int(np.round(len(PMts)/len_traces)),:] #length of this must be the same as that of GCaMP_dF_F
        
    else: # data from FIP 

        print("NPM timestamps not found")
        # PMts = np.load(glob.glob(results_folder + os.sep + "**" + os.sep + "PMts.npy", recursive=True)[0])
        # PMts2 = np.vstack((PMts,np.arange(len(PMts)))).T 
        

    # Timestamp for TTL    
    with open(file_TTLTS) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        TTLts = datatemp[0:,:].astype(np.float32)
        
        
    #%%Sorting NIDAQ-AI channels
    if (len(TTLsignal)/1000) / len(TTLts) == 1:
        print("Num Analog Channel: 1")
        
    elif (len(TTLsignal)/1000) / len(TTLts) == 2:  #this shouldn't happen, though...
        TTLsignal2 = TTLsignal[1::2]
        TTLsignal = TTLsignal[0::2]
        print("Num Analog Channel: 2")
            
    elif (len(TTLsignal)/1000) / len(TTLts) >= 3:
        TTLsignal1 = TTLsignal[0::3]
    
        print("Num Analog Channel: 3")
    else:
        print("Something is wrong with TimeStamps or Analog Recording...")
        
    #%% analoginputs binalize
    TTLsignal = TTLsignal1
    TTLsignal[TTLsignal < 3] = 0
    TTLsignal[TTLsignal >= 3] = 1
    TTLsignal_shift = np.roll(TTLsignal, 1)
    diff = TTLsignal - TTLsignal_shift

    # Sorting
    TTL_p = []
    TTL_l = []

    for ii in range(len(TTLsignal)):
        if diff[ii] == 1:
            for jj in range(120): #Max length:40
                if ii+jj > len(TTLsignal)-1:
                    break
                
                if diff[ii+jj] == -1:
                    TTL_p = np.append(TTL_p, ii) 
                    TTL_l = np.append(TTL_l, jj)
                    break

    ## binalize raw lick signals             
    if 'TTLsignal2' in locals():
            
        TTLsignal2[TTLsignal2 < 0.5] = 0
        TTLsignal2[TTLsignal2 >= 0.5] = 1
        TTLsignal2_shift = np.roll(TTLsignal2, 1)
        diff2 = TTLsignal2 - TTLsignal2_shift
        
        TTL2_p = []
        for ii in range(len(TTLsignal2)):
            if diff2[ii] == 1:
                TTL2_p = np.append(TTL2_p, ii) 

                
    if 'TTLsignal3' in locals():
            
        TTLsignal3[TTLsignal3 < 0.5] = 0
        TTLsignal3[TTLsignal3 >= 0.5] = 1
        TTLsignal3_shift = np.roll(TTLsignal3, 1)
        diff3 = TTLsignal3 - TTLsignal3_shift
        
        TTL3_p = []
        for ii in range(len(TTLsignal3)):
            if diff3[ii] == 1:
                TTL3_p = np.append(TTL3_p, ii)
        
    #%% Alignment between PMT and TTL
    TTL_p_align = []
    TTL_p_align_1k = []
    for ii in range(len(TTL_p)):
        ind_tmp = int(np.ceil(TTL_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
        dec_tmp = TTL_p[ii]/1000 + 1 - np.ceil(TTL_p[ii]/1000)
        if ind_tmp >= len(TTLts):
            break
        ms_target = TTLts[ind_tmp]
        idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))

        residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
        TTL_p_align = np.append(TTL_p_align, idx)
        TTL_p_align_1k = np.append(TTL_p_align_1k, time_seconds[idx] - residual/1000)    
        
    TTL_l_align = TTL_l[0:len(TTL_p_align)] 


    if 'TTL2_p' in locals():
        TTL2_p_align = []
        TTL2_p_align_1k = []
        for ii in range(len(TTL2_p)):
            ind_tmp = int(np.ceil(TTL2_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
            dec_tmp = TTL2_p[ii]/1000 + 1 - np.ceil(TTL2_p[ii]/1000)
            if ind_tmp >= len(TTLts):
                break
            ms_target = TTLts[ind_tmp]
            idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))
            residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
            TTL2_p_align = np.append(TTL2_p_align, idx)
            TTL2_p_align_1k = np.append(TTL2_p_align_1k, time_seconds[idx] - residual/1000)    
            
    if 'TTL3_p' in locals():
        TTL3_p_align = []
        TTL3_p_align_1k = []
        for ii in range(len(TTL3_p)):
            ind_tmp = int(np.ceil(TTL3_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
            dec_tmp = TTL3_p[ii]/1000 + 1 - np.ceil(TTL3_p[ii]/1000)
            if ind_tmp >= len(TTLts):
                break
            ms_target = TTLts[ind_tmp]
            idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))
            residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
            TTL3_p_align = np.append(TTL3_p_align, idx)
            TTL3_p_align_1k = np.append(TTL3_p_align_1k, time_seconds[idx] - residual/1000)   


    return TTLsignal1, TTL_p_align, TTL_l_align, TTL_p, TTL_l 