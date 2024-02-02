#%%
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib
import pylab as plt
import seaborn as sns
import csv
import glob
from sklearn.metrics import r2_score
from matplotlib import gridspec
from scipy.stats import binned_statistic
import sys
#from pynwb import NWBHDF5IO, TimeSeries
import datetime
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)
pd.set_option("display.max_columns", None)

import importlib
from utils import preprocessing, load_session, dict2dataframe, plot_rawexample

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('mode.chained_assignment', None)
# pd.reset_option("mode.chained_assignment")

import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
# signal.signal(signal.SIGALRM, timeout_handler)

# %matplotlib inline

#%% Load dataset files
# os.chdir(os.path.dirname('/home/reca/Dropbox (uwamath)/Bandit_Allen/'))
datafolder = '/data/'
folders_sessions = sorted(glob.glob(datafolder+'*FIP_*'))

#df_sessions = pd.read_pickle('/data/s3_foraging_all_nwb/df_sessions.pkl')      
#df_sessions = df_sessions.sort_values('session_date', ascending=True)
df_sessions = pd.read_pickle('/data/s3_foraging_behavior_npy/df_sessions.pkl')

#% Adding the minimeta details to the sessions
df_minimeta = pd.read_csv('/data/NeuroMod_Foraging_MiniMeta/NeuroMod_Foraging_MiniMeta.csv')
df_minimeta = df_minimeta[['WR_Name', 'LabtracksID', 'Region_0', 'L/R_0', 'Region_1', 'L/R_1', 'Region_2', 'L/R_2', 'Region_3', 'L/R_3', 'G_0', 'G_1', 'R_0', 'R_1']]
df_minimeta = df_minimeta.rename(columns={'WR_Name':'h20','LabtracksID':'subject_id'})
df_sessions = df_sessions.merge(df_minimeta, how='inner', on= 'subject_id')

#% Initialize the dataframe across sessions
merge_behavior = False
df_sessions_new = pd.DataFrame()

#%% Process all the sessions that are attached saving one dataframe across all sessions and one dataframe for all trials in each session
for i_session in np.arange(len(folders_sessions))[:]:
    print(i_session)
    # signal.alarm(90)        
    state = ''
    df_session = pd.DataFrame()
    
    # Create the dataframe for the session
    state = 'creating df_session'
    folder_session = folders_sessions[i_session]
    session_name = folder_session.split('/')[-1]
    subject_id = session_name.split('_')[1]
    date = session_name.split('_')[2]
    time_in_day = session_name.split('_')[3]
    condition_subject_id = df_sessions['subject_id'].astype(str)== subject_id
    condition_date = df_sessions['session_date'].values==pd.Timestamp(date).date()
    df_sessions['hour'] = df_sessions['time_in_day'].apply(lambda x: str(int(x)).rjust(2,'0'))        
    df_session = df_sessions[condition_date & condition_subject_id]
    if len(df_session)==0:
        continue # Continue if the session is not found in the MiniMeta dataframe df_sessions 

    # Initiate the processing of the session
    # try:          
    # Preprocessing stage saving output files in the scratch                
    state = 'preprocessing'
    df_session['date_preprocessed'] = datetime.date.today()
    preprocessing(i_session)

    # Processing stage outputting timestamps for multiple events, recorded traces and trials identity (bit_code)
    state = 'load_session'
    events, traces, bit_code = load_session(i_session, traces_names=['G_0', 'R_0', 'G_1', 'R_1'])         


    # Checking the alignment between the events (go cue) and the trial identity (bit code). This stage can be moved to processing stage
    state = 'alignement bit_code to gocue'                      
    events_match = 0
    N_trials = np.min([len(events['bit_code_times']), len(events['go cue'])])
    if len(bit_code) < len(events['go cue']):   
        values = events['go cue']                
        bin_edges = values[:-1]+np.diff(values)/2
        idxs = np.digitize(events['bit_code_times'], bin_edges)    
        idxs_extra = np.sort(np.array(list(set(np.arange(len(events['go cue']))).difference(idxs))))
        events['bit_code_times'] = np.insert(events['bit_code_times'],idxs_extra-np.arange(len(idxs_extra)),np.nan)
        bit_code = np.insert(bit_code,idxs_extra-np.arange(len(idxs_extra)),np.nan)            

        idxs_extra1, = np.where(events['go cue'][idxs]<events['bit_code_times'][idxs])
        idxs_extra2, = np.where(events['go cue'][idxs[1:]]<events['bit_code_times'][idxs[:-1]])
        idxs_extra12 = np.concatenate([idxs_extra1,idxs_extra2])
        events['bit_code_times'][idxs_extra12] = np.nan
        bit_code[idxs_extra12] = np.nan

        events_match = 1
        print('bit code < #trials')

    if len(bit_code) > len(events['go cue']):        
        values = events['bit_code_times']                
        bin_edges = values[:-1]+np.diff(values)/2
        idxs = np.digitize(events['go cue'], bin_edges)
        
        events['bit_code_times'] = events['bit_code_times'][idxs]
        bit_code = np.array(bit_code)[idxs]
        
        idxs_extra1, = np.where(events['go cue']<events['bit_code_times'])
        idxs_extra2, = np.where(events['go cue'][1:]<events['bit_code_times'][:-1])
        idxs_extra12 = np.concatenate([idxs_extra1, idxs_extra2])
        events['bit_code_times'][idxs_extra12] = np.nan
        bit_code[idxs_extra12] = np.nan

        events_match = 2
        print('#trials < bit code')        
    df_session['events_match'] = events_match   
    
    # From now on the indicator for each session will be subject_id and session date. Here we verify if multiple sessions were recorded in the same day
    if len(df_session) > 1:            
        print('ambiguous identification')
        condition_hour = df_sessions['hour']==time_in_day.split('-')[0]
        df_session = df_sessions[condition_hour & condition_date & condition_subject_id]                    

    # Extract results of processing
    trace_times = traces.pop('time_seconds')
    traces_names = list(traces.keys())
    traces_renamed = {df_session[key].values[0]:traces[key] for key in traces.keys()}
    events_names = list(events.keys())

    files = glob.glob('/results/'+folder_session.split('/')[-1]+'/*')
    for f in files:
        os.remove(f)

    # Create output dataframe across trials for the session
    dfnew = dict2dataframe(events, traces, trace_times, bin_size=0.05)
    dfnew['session'] = session_name
    dfnew['i_session'] = i_session
    dfnew['time_in_day'] = time_in_day
    dfnew['session_date'] = date
    dfnew['subject_id'] = subject_id
    dfnew['bit_code'] = bit_code                
            
    # Merge behavior from Han's dataset. This is now done downstream in the output of this capsule.
    if merge_behavior:
        trainer = df_session['h2o'].values[0]
        date = str(df_session['session_date'].values[0]).replace('-','')
        i_ses = str(df_session['session'].values[0])
        session_name = trainer +'_'+date+'_'+i_ses
        filename_han = glob.glob('/data/s3_foraging_all_nwb/' + trainer + '/' + session_name + '.nwb')
        state = 'loading Han database'       
        io = NWBHDF5IO(filename_han[0], mode='r')
        nwb = io.read()
        df_session_vals = nwb.trials.to_dataframe()

        choices_values = df_session_vals['choice'].map({'right':1, 'left':0}).astype(float).values
        rewards_values = df_session_vals['outcome'].map({'miss':0, 'hit':1}).astype(float).values

        df_ses2fit = df_session
        df_ses2fit = df_ses2fit[['session_date', 'h2o', 'session', 'subject_id']]
        df_ses2fit['choice_history'] = [choices_values]
        df_ses2fit = df_ses2fit.explode('choice_history')
        df_ses2fit['reward_history'] = rewards_values
        df_ses2fit['right_reward_'] = df_session_vals['right_reward_prob'].values
        df_ses2fit['p_reward_left'] = df_session_vals['left_reward_prob'].values
        df_ses2fit['bit_code'] = df_session_vals['trial_bit_code'].values
        if len(df_ses2fit) == len(dfnew):
            Ntrials_match = True
        else:
            Ntrials_match = False        
        df_new = dfnew.merge(df_ses2fit, how='inner', left_on='bit_code', right_on='bit_code')     
    else:            
        Ntrials_match = np.nan
        df_new = dfnew            
    df_session['Ntrials_match'] = Ntrials_match        
        
    # Save the trials structure
    # state = 'saving file'       
    # resultsfolder='/results/'+folder_session.split('/')[-1]
    # df_new.to_pickle(resultsfolder+'/'+'df_'+folder_session.split('/')[-1]+'.pk')               
    print("Succesfully processed: " + folders_sessions[i_session])    
        
    # except:    
    #     print("Failed to process: " + folders_sessions[i_session])
    #     files = glob.glob('/results/'+folder_session.split('/')[-1]+'/*')
    #     for f in files:
    #         os.remove(f)        

    # else:                
        # signal.alarm(0)        

    # Concatenate all sessions
    df_session['state'] = state
    df_sessions_new = pd.concat([df_sessions_new, df_session], axis=0)
        
# Save dataframe across processed sessions
# df_sessions_new.to_pickle('/results/df_sessions.pk')

#%%