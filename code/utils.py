#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import pandas as pd
import seaborn as sns
import glob
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import binned_statistic, linregress
from scipy.optimize import curve_fit, minimize
import os
import csv
from matplotlib.patches import ConnectionPatch

#%%
def preprocessing(i_session, datafolder='/data', resultsfolder='/results'):
    print("starting preprocessing")
    
    folders_sessions = sorted(glob.glob(datafolder+os.sep+'*'))
    folder_session = folders_sessions[i_session]
    AnalDir = folder_session+'/FIP'
    resultsfolder=resultsfolder+os.sep+folder_session.split('/')[-1]
    isExist = os.path.exists(resultsfolder)
    if not isExist:   
        os.makedirs(resultsfolder)
    # os.mkdir('/results/'+folder_session.split('/')[-1]) 

    nFibers = 2
    nColor = 3
    sampling_rate = 20 #individual channel (not total)
    nFrame2cut = 100  #crop initial n frames
    b_percentile = 0.70 #To calculare F0, median of bottom x%

    
    #BiExpFitIni = [1,1e-3,5,1e-3,5]
    BiExpFitIni = [1,1e-3,1,1e-3,1]  #currentlu not used

    if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
        # print('preprocessing Neurophotometrics Data')
        file1 = glob.glob(AnalDir + os.sep + "L415*")[0]
        file2 = glob.glob(AnalDir + os.sep + "L470*")[0]
        file3 = glob.glob(AnalDir + os.sep + "L560*")[0]
    elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
        # print('preprocessing FIP Data')
        file1 = glob.glob(AnalDir + os.sep + "FIP_DataIso*")[0]
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
    
    #%% Data sort # 1,2:Time-Frame info; ROI0:3;ROI1:4,ROI2:5,ROI3:6;...
    #Data_Fiber1iso = data1[:,3]
    #Data_Fiber1G = data2[:,3]
    #Data_Fiber1R = data3[:,5]
    
    #Data_Fiber2iso = data1[:,4]
    #Data_Fiber2G = data2[:,4]
    #Data_Fiber2R = data3[:,6]
    #%% from 220609-, ROI0:8;ROI1:9,ROI2:10,ROI3:11;

    if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
        if data1.shape[1] > 7:
            Data_Fiber1iso = data1[:,8]
            Data_Fiber1G = data2[:,8]
            Data_Fiber1R = data3[:,10]
            Data_Fiber2iso = data1[:,9]
            Data_Fiber2G = data2[:,9]
            Data_Fiber2R = data3[:,11]
        else:
            Data_Fiber1iso = data1[:,3]
            Data_Fiber1G = data2[:,3]
            Data_Fiber1R = data3[:,5]
            Data_Fiber2iso = data1[:,4]
            Data_Fiber2G = data2[:,4]
            Data_Fiber2R = data3[:,6]
        file_TS = glob.glob(AnalDir + os.sep + "TimeStamp_*")[0]
        with open(file_TS) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            PMts = datatemp[0:, :].astype(np.float32)
    elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:        
        Data_Fiber1iso = data1[:,1]
        Data_Fiber1G = data2[:,1]
        Data_Fiber1R = data3[:,1]
        Data_Fiber2iso = data1[:,2]
        Data_Fiber2G = data2[:,2]
        Data_Fiber2R = data3[:,2]
        PMts = data2[:,0]


    #%% From here to be multiplexed

    #cropping
    G1_raw = Data_Fiber1G[nFrame2cut:]
    G2_raw = Data_Fiber2G[nFrame2cut:]
    R1_raw = Data_Fiber1R[nFrame2cut:]
    R2_raw = Data_Fiber2R[nFrame2cut:]
    Ctrl1_raw = Data_Fiber1iso[nFrame2cut:]
    Ctrl2_raw = Data_Fiber2iso[nFrame2cut:]

    time_seconds = np.arange(len(G1_raw)) /sampling_rate 

    

    #%% Median filtering to remove electrical artifact.
    kernelSize=1
    G1_denoised = medfilt(G1_raw, kernel_size=kernelSize)
    G2_denoised = medfilt(G2_raw, kernel_size=kernelSize)
    R1_denoised = medfilt(R1_raw, kernel_size=kernelSize)
    R2_denoised = medfilt(R2_raw, kernel_size=kernelSize)
    Ctrl1_denoised = medfilt(Ctrl1_raw, kernel_size=5)
    Ctrl2_denoised = medfilt(Ctrl2_raw, kernel_size=5)
    
    # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
    b,a = butter(2, 9, btype='low', fs=sampling_rate)
    G1_denoised = filtfilt(b,a, G1_denoised)
    G2_denoised = filtfilt(b,a, G2_denoised)
    R1_denoised = filtfilt(b,a, R1_denoised)
    R2_denoised = filtfilt(b,a, R2_denoised)
    Ctrl1_denoised = filtfilt(b,a, Ctrl1_denoised)
    Ctrl2_denoised = filtfilt(b,a, Ctrl2_denoised)
    # plt.legend()



    #%% Photobleaching correction by LowCut
    '''
    b,a = butter(2, 0.05, btype='high', fs=sampling_rate)
    G1_highpass = filtfilt(b,a, G1_denoised, padtype='even')
    G2_highpass = filtfilt(b,a, G2_denoised, padtype='even')
    R1_highpass = filtfilt(b,a, R1_denoised, padtype='even')
    R2_highpass = filtfilt(b,a, R2_denoised, padtype='even')
    Ctrl1_highpass = filtfilt(b,a, Ctrl1_denoised, padtype='even')
    Ctrl2_highpass = filtfilt(b,a, Ctrl2_denoised, padtype='even')
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(time_seconds, G1_highpass,'g', label='G1 highpass')
    plt.plot(time_seconds, R1_highpass,'r', label='R1 highpass')
    plt.plot(time_seconds, Ctrl1_highpass,'b', label='iso1 highpass')
    plt.subplot(1,2,2)
    plt.plot(time_seconds, G2_highpass,'g', label='G2 highpass')
    plt.plot(time_seconds, R2_highpass,'r', label='R2 highpass')
    plt.plot(time_seconds, Ctrl2_highpass,'b', label='iso2 highpass')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CMOS Signal')
    plt.title('Bleaching correction by highpass filtering')
    plt.legend();
    '''
    #%% Bi-exponential curve fit.
    '''
    #def exp_func(x, a, b, c):
    #   return a*np.exp(-b*x) + c
    BiExpFitIni = [1,1e-3,5,1e-3,5]
    def biexpF(x, a, b, c, d, e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e
    # Fit curve to signals.
    G1_parms, parm_cov1 = curve_fit(biexpF, time_seconds, G1_denoised, p0=BiExpFitIni,maxfev=5000)
    G1_expfit = biexpF(time_seconds, *G1_parms)
    G2_parms, parm_cov2 = curve_fit(biexpF, time_seconds, G2_denoised, p0=BiExpFitIni,maxfev=5000)
    G2_expfit = biexpF(time_seconds, *G2_parms)
    R1_parms, parm_cov1 = curve_fit(biexpF, time_seconds, R1_denoised, p0=BiExpFitIni,maxfev=5000)
    R1_expfit = biexpF(time_seconds, *R1_parms)
    R2_parms, parm_cov2 = curve_fit(biexpF, time_seconds, R2_denoised, p0=BiExpFitIni,maxfev=5000)
    R2_expfit = biexpF(time_seconds, *R2_parms)
    # Fit curve to ctrl.
    Ctrl1_parms, parm_cov = curve_fit(biexpF, time_seconds, Ctrl1_denoised, p0=BiExpFitIni,maxfev=5000)
    Ctrl1_expfit = biexpF(time_seconds, *Ctrl1_parms)
    Ctrl2_parms, parm_cov = curve_fit(biexpF, time_seconds, Ctrl2_denoised, p0=BiExpFitIni,maxfev=5000)
    Ctrl2_expfit = biexpF(time_seconds, *Ctrl2_parms)
    '''
    #%%
    # Fit 4th order polynomial to signals.
    coefs_G1 = np.polyfit(time_seconds, G1_denoised, deg=4)
    G1_polyfit = np.polyval(coefs_G1, time_seconds)
    coefs_G2 = np.polyfit(time_seconds, G2_denoised, deg=4)
    G2_polyfit = np.polyval(coefs_G2, time_seconds)
    coefs_R1 = np.polyfit(time_seconds, R1_denoised, deg=4)
    R1_polyfit = np.polyval(coefs_R1, time_seconds)
    coefs_R2 = np.polyfit(time_seconds, R2_denoised, deg=4)
    R2_polyfit = np.polyval(coefs_R2, time_seconds)

    # Fit 4th order polynomial to Ctrl.
    coefs_Ctrl1 = np.polyfit(time_seconds, Ctrl1_denoised, deg=4)
    Ctrl1_polyfit = np.polyval(coefs_Ctrl1, time_seconds)
    coefs_Ctrl2 = np.polyfit(time_seconds, Ctrl2_denoised, deg=4)
    Ctrl2_polyfit = np.polyval(coefs_Ctrl2, time_seconds)


    G1_es = G1_denoised - G1_polyfit
    G2_es = G2_denoised - G2_polyfit
    R1_es = R1_denoised - R1_polyfit
    R2_es = R2_denoised - R2_polyfit
    Ctrl1_es = Ctrl1_denoised - Ctrl1_polyfit
    Ctrl2_es = Ctrl2_denoised - Ctrl2_polyfit

    #%%Additional LowCut
    '''
    b,a = butter(2, 0.1, btype='high', fs=sampling_rate)
    G1_es2 = filtfilt(b,a, G1_es, padtype='even')
    G2_es2 = filtfilt(b,a, G2_es, padtype='even')
    R1_es2 = filtfilt(b,a, R1_es, padtype='even')
    R2_es2 = filtfilt(b,a, R2_es, padtype='even')
    Ctrl1_es2 = filtfilt(b,a, Ctrl1_es, padtype='even')
    Ctrl2_es2 = filtfilt(b,a, Ctrl2_es, padtype='even')
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(time_seconds, G1_es2,'g', label='G1 highpass')
    plt.plot(time_seconds, R1_es2,'r', label='R1 highpass')
    plt.plot(time_seconds, Ctrl1_es2,'b', label='iso1 highpass')
    plt.subplot(1,2,2)
    plt.plot(time_seconds, G2_es2,'g', label='G2 highpass')
    plt.plot(time_seconds, R2_es2,'r', label='R2 highpass')
    plt.plot(time_seconds, Ctrl2_es2,'b', label='iso2 highpass')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CMOS Signal')
    plt.title('Bleaching correction by highpass filtering')
    plt.legend();
    '''

    #%% Motion correction using iso
    slopeG1, interceptG1, r_valueG1, p_valueG1, std_errG1 = linregress(x=Ctrl1_es, y=G1_es)
    slopeG2, interceptG2, r_valueG2, p_valueG2, std_errG2 = linregress(x=Ctrl2_es, y=G2_es)
    slopeR1, interceptR1, r_valueR1, p_valueR1, std_errR1 = linregress(x=Ctrl1_es, y=R1_es)
    slopeR2, interceptR2, r_valueR2, p_valueR2, std_errR2 = linregress(x=Ctrl2_es, y=R2_es)

    

    #% motion corrected
    G1_est_motion = interceptG1 + slopeG1 * Ctrl1_es
    G1_corrected = G1_es - G1_est_motion
    G2_est_motion = interceptG2 + slopeG2 * Ctrl2_es
    G2_corrected = G2_es - G2_est_motion

    R1_est_motion = interceptR1 + slopeR1 * Ctrl1_es
    R1_corrected = R1_es - G1_est_motion
    R2_est_motion = interceptR2 + slopeR2 * Ctrl2_es
    R2_corrected = R2_es - R2_est_motion



    #%% dF/F using sliding baseline
    b,a = butter(2, 0.0001, btype='low', fs=sampling_rate)
    G1_baseline = filtfilt(b,a, G1_denoised, padtype='even')
    G2_baseline = filtfilt(b,a, G2_denoised, padtype='even')
    R1_baseline = filtfilt(b,a, R1_denoised, padtype='even')
    R2_baseline = filtfilt(b,a, R2_denoised, padtype='even')

    Ctrl1_baseline = filtfilt(b,a, Ctrl1_denoised, padtype='even')
    Ctrl2_baseline = filtfilt(b,a, Ctrl2_denoised, padtype='even')

    #G1_dF_F = G1_corrected/G1_baseline
    G1_dF_F = G1_es/G1_baseline
    sort = np.sort(G1_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    G1_dF_F = G1_dF_F - b_median

    #G2_dF_F = G2_corrected/G2_baseline
    G2_dF_F = G2_es/G2_baseline
    sort = np.sort(G2_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    G2_dF_F = G2_dF_F - b_median

    #R1_dF_F = R1_corrected/R1_baseline
    R1_dF_F = R1_es/R1_baseline
    sort = np.sort(R1_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    R1_dF_F = R1_dF_F - b_median

    #R2_dF_F = R2_corrected/R2_baseline
    R2_dF_F = R2_es/R2_baseline
    sort = np.sort(R2_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    R2_dF_F = R2_dF_F - b_median

    Ctrl1_dF_F = Ctrl1_es/Ctrl1_baseline
    sort = np.sort(Ctrl1_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    Ctrl1_dF_F = Ctrl1_dF_F - b_median

    Ctrl2_dF_F = Ctrl2_es/Ctrl2_baseline
    sort = np.sort(Ctrl2_dF_F)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    Ctrl2_dF_F = Ctrl2_dF_F - b_median


    #%%
    G1_dF_F = np.append(np.ones([nFrame2cut,1])*G1_dF_F[0],G1_dF_F)
    G2_dF_F = np.append(np.ones([nFrame2cut,1])*G2_dF_F[0],G2_dF_F)
    R1_dF_F = np.append(np.ones([nFrame2cut,1])*R1_dF_F[0],R1_dF_F)
    R2_dF_F = np.append(np.ones([nFrame2cut,1])*R2_dF_F[0],R2_dF_F)

    Ctrl1_dF_F = np.append(np.ones([nFrame2cut,1])*Ctrl1_dF_F[0],Ctrl1_dF_F)
    Ctrl2_dF_F = np.append(np.ones([nFrame2cut,1])*Ctrl2_dF_F[0],Ctrl2_dF_F)

    #%%
    time_seconds = np.arange(len(G1_dF_F)) /sampling_rate 

    #%% Save
    print("line 324")
    np.save(resultsfolder +os.sep+ "G1_dF_F", G1_dF_F)
    np.save(resultsfolder +os.sep+  "G2_dF_F", G2_dF_F)
    np.save(resultsfolder +os.sep+  "R1_dF_F", R1_dF_F)
    np.save(resultsfolder +os.sep+  "R2_dF_F", R2_dF_F)
    np.save(resultsfolder +os.sep+  "Ctrl1_dF_F", Ctrl1_dF_F)
    np.save(resultsfolder +os.sep+  "Ctrl2_dF_F", Ctrl2_dF_F)
    np.save(resultsfolder +os.sep+  "PMts", PMts)
    print("line 332")

    return time_seconds



def load_session(i_session, datafolder='/data', resultsfolder='/results', traces_names=['G1', 'R1', 'G2', 'R2']):
    #% Locate dataset files
    # os.chdir(os.path.dirname('/home/reca/Dropbox (uwamath)/Bandit_Allen/'))
    # datafolder = 'data/'
    folders_sessions = sorted(glob.glob(datafolder+os.sep+'*'))

    #% Load files for specific session
    # AnalDir = r"/home/reca/Dropbox (uwamath)/Bandit_Allen/fpNPM_KH-FB31_2022-09-01_09-32-00"
    # i_session = 0

    if i_session > len(folders_sessions):
        return None

    folder_session = folders_sessions[i_session]

    session = folder_session.split('/')[-1]
    preprocessing_folder = resultsfolder + os.sep + session

    folder_session = folders_sessions[i_session]+'/FIP'
    FlagNoRawLick = 0

    Trace0 = np.load(glob.glob(preprocessing_folder + os.sep + "G1_dF_F.npy")[0])
    Trace1 = np.load(glob.glob(preprocessing_folder + os.sep + "R1_dF_F.npy")[0])
    Trace2 = np.load(glob.glob(preprocessing_folder + os.sep + "G2_dF_F.npy")[0])
    Trace3 = np.load(glob.glob(preprocessing_folder + os.sep + "R2_dF_F.npy")[0])

    Traces = np.vstack((Trace0, Trace1, Trace2, Trace3))
    Traces = Traces.T
    time_seconds = np.arange(len(Traces)) /20

    TTLsignal = np.fromfile(glob.glob(folder_session + os.sep + "TTL_20*")[0])
    file_TTLTS = glob.glob(folder_session + os.sep + "TTL_TS*")[0]

    PMts = np.load(glob.glob(preprocessing_folder + os.sep + "PMts.npy")[0])
    if glob.glob(folder_session + os.sep + "TimeStamp_*") != []:  # data from NPM
        # file_TS = glob.glob(folder_session + os.sep + "TimeStamp_*")[0]
        # with open(file_TS) as f:
        #     reader = csv.reader(f)
        #     datatemp = np.array([row for row in reader])
        #     PMts = datatemp[0:, :].astype(np.float32)
        PMts2 = PMts[0:len(PMts):int(np.round(len(PMts) / len(Traces))),:]  # length of this must be the same as that of GCaMP_dF_F
    else:  # data from FIP    
        PMts2 = np.vstack((PMts, np.arange(len(PMts)))).T

# There are 3 TTL signals, the binary code developed from Han, the second and third channels are the right and left licks.

    # Timestamp for TTL
    with open(file_TTLTS) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        TTLts = datatemp[0:, :].astype(np.float32)

    #% Sorting NIDAQ-AI channels
    if (len(TTLsignal) / 1000) / len(TTLts) == 1: # dividing by 1000 gets ms scale from 1.kHz
        print("Num Analog Channel: 1")
    elif (len(TTLsignal) / 1000) / len(TTLts) == 2:
        TTLsignal2 = TTLsignal[1::2]
        TTLsignal = TTLsignal[0::2]
        # print("Num Analog Channel: 2")
    elif (len(TTLsignal) / 1000) / len(TTLts) >= 3:
        TTLsignal1 = TTLsignal[0::3]
        if FlagNoRawLick == 0:
            TTLsignal2 = TTLsignal[1::3]
            TTLsignal3 = TTLsignal[2::3]
        # print("Num Analog Channel: 3")
    else:
        print("Something is wrong with TimeStamps or Analog Recording...")

    #% analog inputs binarize
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
            for jj in range(120):  # Max length:40
                if ii + jj > len(TTLsignal) - 1:
                    break
                if diff[ii + jj] == -1:
                    TTL_p = np.append(TTL_p, ii) #this is the onset = timestamp
                    TTL_l = np.append(TTL_l, jj) # this the length of the TTL pulse which is Han's code
                    break

    values = np.array([ 1.,  2.,  3., 10., 20., 30., 40.])
    bin_edges = np.concatenate([values[:-1]+np.diff(values)/2, [values[-1]+10]])
    idxs = np.digitize(TTL_l, bin_edges)
    TTL_l = values[idxs]

    #% binarize raw lick signals
    if 'TTLsignal2' in locals():
        if FlagNoRawLick == 1:
            TTLsignal2[:] = 0
        TTLsignal2[TTLsignal2 < 0.5] = 0
        TTLsignal2[TTLsignal2 >= 0.5] = 1
        TTLsignal2_shift = np.roll(TTLsignal2, 1)
        diff2 = TTLsignal2 - TTLsignal2_shift
        TTL2_p = []
        for ii in range(len(TTLsignal2)):
            if diff2[ii] == 1:
                TTL2_p = np.append(TTL2_p, ii)

    if 'TTLsignal3' in locals():
        if FlagNoRawLick == 1:
            TTLsignal3[:] = 0
        TTLsignal3[TTLsignal3 < 0.5] = 0
        TTLsignal3[TTLsignal3 >= 0.5] = 1
        TTLsignal3_shift = np.roll(TTLsignal3, 1)
        diff3 = TTLsignal3 - TTLsignal3_shift
        TTL3_p = []
        for ii in range(len(TTLsignal3)):
            if diff3[ii] == 1:
                TTL3_p = np.append(TTL3_p, ii)

    #% Alignment between PMT (photometry recorded at 20Hz) and TTL (sampled at 1kHz)
    TTL_p_align = []
    TTL_p_align_1k = []

    for ii in range(len(TTL_p)):
        ind_tmp = int(np.ceil(TTL_p[ii] / 1000) - 2)  # consider NIDAQ buffer 1s (1000samples@1kHz) # this 2 comes from bonsai alignment
        dec_tmp = TTL_p[ii] / 1000 + 1 - np.ceil(TTL_p[ii] / 1000)
        if ind_tmp >= len(TTLts):
            break
        ms_target = TTLts[ind_tmp]
        idx = int(np.argmin(np.abs(np.array(PMts2[:len(time_seconds), 0]) - ms_target - dec_tmp * 1000)))
        residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
        TTL_p_align = np.append(TTL_p_align, idx)
        TTL_p_align_1k = np.append(TTL_p_align_1k, time_seconds[idx] - residual/1000)
        
    TTL_l_align = TTL_l[0:len(TTL_p_align)]
    if 'TTL2_p' in locals():
        TTL2_p_align = []
        TTL2_p_raw = []
        TTL2_p_align_1k = []
        for ii in range(len(TTL2_p)):
            ind_tmp = int(np.ceil(TTL2_p[ii] / 1000) - 2)  # consider NIDAQ buffer 1s (1000samples@1kHz)
            dec_tmp = TTL2_p[ii] / 1000 + 1 - np.ceil(TTL2_p[ii] / 1000)
            if ind_tmp >= len(TTLts):
                break
            ms_target = TTLts[ind_tmp]
            idx = int(np.argmin(np.abs(np.array(PMts2[:len(time_seconds), 0]) - ms_target - dec_tmp * 1000)))
            residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
            TTL2_p_align = np.append(TTL2_p_align, idx)            
            TTL2_p_align_1k = np.append(TTL2_p_align_1k, time_seconds[idx] - residual/1000)

    if 'TTL3_p' in locals():
        TTL3_p_align = []
        TTL3_p_raw = []
        TTL3_p_align_1k = []
        for ii in range(len(TTL3_p)):
            ind_tmp = int(np.ceil(TTL3_p[ii] / 1000) - 2)  # consider NIDAQ buffer 1s (1000samples@1kHz)
            dec_tmp = TTL3_p[ii] / 1000 + 1 - np.ceil(TTL3_p[ii] / 1000)
            if ind_tmp >= len(TTLts):
                break
            ms_target = TTLts[ind_tmp]
            idx = int(np.argmin(np.abs(np.array(PMts2[:len(time_seconds), 0]) - ms_target - dec_tmp * 1000)))
            residual = np.array(PMts2[idx,0]) - ms_target - dec_tmp*1000
            TTL3_p_align = np.append(TTL3_p_align, idx)           
            TTL3_p_align_1k = np.append(TTL3_p_align_1k, time_seconds[idx] - residual/1000)

    #% Rewarded Unrewarded L/R trials, Left has a code of 2 and right has a code of 3 in the next snippet

    RewardedL = []
    UnRewardedL = []
    RewardedR = []
    UnRewardedR = []
    for ii in range(len(TTL_l_align) - 1):
        if TTL_l_align[ii] == 2 and TTL_l_align[ii + 1] == 30:  # 30:reward, #40: ITI start
            RewardedL = np.append(RewardedL, ii)
        if TTL_l_align[ii] == 2 and TTL_l_align[ii + 1] == 40:
            UnRewardedL = np.append(UnRewardedL, ii)
        if TTL_l_align[ii] == 3 and TTL_l_align[ii + 1] == 30:
            RewardedR = np.append(RewardedR, ii)
        if TTL_l_align[ii] == 3 and TTL_l_align[ii + 1] == 40:
            UnRewardedR = np.append(UnRewardedR, ii)
    UnRewarded = np.union1d(UnRewardedL, UnRewardedR)
    Rewarded = np.union1d(RewardedL, RewardedR)

    Ignored = []

    for ii in range(len(TTL_l_align) - 1):
        if TTL_l_align[ii] == 1 and TTL_l_align[ii + 1] == 40:  # 1:GoCue, #40: ITI start
            Ignored = np.append(Ignored, ii)

    #% Barcode Decode (220626 updated)
    BarcodeP = TTL_p[TTL_l == 20] # this 20 is the trial initial barcode length for each bit
    BarcodeBin = np.zeros((len(BarcodeP), 20))

    BarcodeP_1k = TTL_p_align_1k[TTL_l == 20]    
  
    for ii in range(len(BarcodeP)):
        for jj in range(20):
            BarcodeBin[ii, jj] = TTLsignal1[int(BarcodeP[ii]) + 30 + 20 * jj + 5]  # checking the middle of 10ms windows
    BarChar = []

    for ii in range(len(BarcodeP)):
        temp = BarcodeBin[ii].astype(int)
        temp2 = ''
        for jj in range(20):
            temp2 = temp2 + str(temp[jj])
        BarChar.append(temp2)
        del temp, temp2

    #% Ca, Behavior Overview
    time_seconds = np.arange(len(Traces)) / 20
    events_names = ['go cue', 'choice L', 'choice R', 'Reward', 'Lick L (raw)', 'Lick R (raw)']
    temp = [1, 2, 3, 30]
    events = {events_names[i]: TTL_p_align[TTL_l_align == temp[i]] / 20 for i in range(len(temp))}
    events.update({'Lick L (raw)': TTL2_p_align / 20})
    events.update({'Lick R (raw)': TTL3_p_align / 20})
    events_1k = {events_names[i]: TTL_p_align_1k[TTL_l_align == temp[i]] for i in range(len(temp))}
    events_1k.update({'Lick L (raw)': TTL2_p_align_1k})
    events_1k.update({'Lick R (raw)': TTL3_p_align_1k})
    events_1k.update({'bit_code_times': BarcodeP_1k})

    # if session == 'KH_FB42':
        # traces_names = ['5-HT', 'DA', 'NE', 'ACh']
    # else:
        # traces_names = ['NE', 'ACh', '5-HT', 'DA']

    traces = {trace_name:Traces[:,i_trace] for i_trace, trace_name in enumerate(traces_names)}
    traces['time_seconds']= time_seconds

    # if session in ['KH_FB43', 'KH_FB52']:
        # traces.pop('5-HT')
        # traces.pop('NE')
        # traces.pop('ACh')

    bit_code = BarChar
    return events_1k, traces, bit_code

#%%
def plot_rawexample(events, traces, trace_times, time_span=[0,600], events_colors = None, session_name='', save=True, plot=False, resultsfolder='/results', tag=''):
    # time_span = [0,600]
#%
# time_span = [0,trace_times.max()]
# save=True
# plot=True
# resultsfolder = '/results'
    traces_names = list(traces.keys())
    events_names = list(events.keys())
    events2plot = {event_name:event_times[(event_times > time_span[0]) & (event_times < time_span[1])] for event_name, event_times in events.items()}
    traces2plot = {trace_name:trace[(trace_times > time_span[0]) & (trace_times < time_span[1])] for trace_name, trace in traces.items()}
    trace_times2plot = trace_times[(trace_times > time_span[0]) & (trace_times < time_span[1])]
    events2plot['Lick L (raw)'] = list(set(events2plot['Lick L (raw)']).difference(set(events2plot['choice L'])))
    events2plot['Lick R (raw)'] = list(set(events2plot['Lick R (raw)']).difference(set(events2plot['choice R'])))
    #%
    if events_colors == None:
        events_colors = [matplotlib.cm.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    plt.clf()
    fig, axs = plt.subplots(5,1, figsize=(200, 6))
    fig = plt.figure( figsize=(200, 6))
    gs = gridspec.GridSpec(6, 1, height_ratios=[.2,.3,1,1,1,1], wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.01, right=0.99)

    # offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    # trans = plt.gca().transData
    # import matplotlib.transforms as transforms

    ax = plt.subplot(gs[0])
    ax.set_axis_off()
    ax.margins(x=0)
    ax.set_xlim(time_span)
    ax.set_ylim([0,5])
    min_last = np.min(np.diff(events2plot['go cue']))
    for i_event, event in enumerate(events_names[:4]):
        for event_time in events2plot[event]:
            ax.plot([event_time, event_time+min_last], [i_event+1, i_event+1], color=events_colors[i_event], linewidth=4., zorder=0, clip_on=False)

    ax = plt.subplot(gs[1])
    for i_event, event in enumerate(events_names):
        event_times = events2plot[event]
        ax.scatter(event_times, np.ones_like(event_times) * i_event * 0.3, s=20, c=events_colors[i_event], marker='|', label=event)#, transform=trans+offset(0))
        ax.set_axis_off()
        ax.margins(x=0)
    ax.set_xlim(time_span)

    for i_trace, trace_name in enumerate(traces_names):
        ax = plt.subplot(gs[i_trace+2])
        ax.margins(x=0)
        ax.plot(trace_times2plot, traces2plot[trace_name] * 100, linewidth=1., color='black')
        ax.plot(trace_times2plot, np.zeros(len(trace_times2plot)), '--', color='gray', linewidth=0.4)
        ax.set_axis_off()
        ax.set_ylabel('dF/F (%)')
        ax.set_title('dF/F  ROI# ' + trace_name)
        ax_min, ax_max = ax.get_ylim()
        for i_event, event in enumerate(events_names[:4]):
            for event_time in events2plot[event]:
                # ax.plot([event_time, event_time], [ax_min*.5, ax_max*.5], color=events_colors[i_event], linewidth=0.4, zorder=0, clip_on=False)
                if i_trace == 0:
                    con = ConnectionPatch(xyA=[event_time, ax_max*.5], xyB=[event_time, ax_min*.5], coordsA="data", coordsB="data", axesA=plt.subplot(gs[2]), axesB=plt.subplot(gs[-1]), color=events_colors[i_event], linewidth=1., zorder=0, clip_on=False)
                    ax.add_artist(con)
        ax.set_ylim([ax_min*.75, ax_max*.75])
        ax.set_xlim(time_span)

    ax.set_xlabel('Time (seconds)')
    for i_event, event in enumerate(events_names):
        ax.scatter(time_span[0], time_span[0], s=8., c=events_colors[i_event], label=event)        

    plt.subplots_adjust(wspace=-0.0, hspace=-0.0)
    sns.despine()
    plt.legend()
    # plt.tight_layout()
    if save:
        plt.savefig(resultsfolder +os.sep + session_name + os.sep + 'fig_RawOverview_'+tag+'.pdf')
    if plot:
        plt.show()
    plt.close()

#%%
def dict2dataframe(events, traces, trace_times, bin_size=0.05):
    traces_names = list(traces.keys())
    events_names = list(events.keys())
    N_trials = len(events['go cue'])
    bins_trials = np.append(events['go cue'], events['go cue'][-1]+np.diff(events['go cue']).max())
    trials = np.arange(N_trials)
    gocues = events['go cue']
    choices = np.concatenate([events['choice L'], events['choice R']])
    # licks = np.concatenate([events['Lick L (raw)'], events['Lick R (raw)']])
    reward_times = binned_statistic(events['Reward'], events['Reward'], statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]
    choice_times = binned_statistic(choices, choices, statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]
    trials_with_reward = binned_statistic(events['Reward'], events['Reward'], statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]>0
    # trials_with_choice = binned_statistic(choices, choices, statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]>0
    choiceL = binned_statistic(events['choice L'], events['choice L'], statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]>0
    choiceR = binned_statistic(events['choice R'], events['choice R'], statistic='min', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]>0
    licksL = binned_statistic(events['Lick L (raw)'], events['Lick L (raw)'], statistic='count', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]
    licksR = binned_statistic(events['Lick R (raw)'], events['Lick R (raw)'], statistic='count', bins=bins_trials, range=[bins_trials.min(), bins_trials.max()])[0]

    df_new = pd.DataFrame({'trial':trials, 'reward':trials_with_reward, 'choiceR':choiceR, 'choiceL':choiceL})
    df_new['choice'] = np.nan * np.zeros_like(trials)
    df_new['choice'][df_new['choiceL']] = 'L'
    df_new['choice'][df_new['choiceR']] = 'R'
    df_new['go_cue_absolute_time'] = gocues
    df_new['go_cue'] = 0
    df_new['choice_time'] = choice_times - gocues
    df_new['reaction_time'] = choice_times - gocues
    df_new['reward_time'] = reward_times - gocues
    df_new['licks L'] = licksL
    df_new['licks R'] = licksR
    df_new['bit_code_time'] = events['bit_code_times']

    traces_trials = {'bins_trial':[],'Lick L (raw)':[], 'Lick R (raw)':[]}
    trial_offset = [-1.,0]
    for i_trace, trace_name in enumerate(traces_names):
        traces_trials.update({trace_name:[]})
        for i_trial in range(N_trials):
            bins_trial = np.arange(bins_trials[i_trial]+trial_offset[0], bins_trials[i_trial+1]+trial_offset[1], bin_size)
            data_mean =  binned_statistic(trace_times, traces[trace_name], statistic='mean', bins=bins_trial, range=[bins_trial.min(), bins_trial.max()])[0]
            traces_trials[trace_name].extend([data_mean])
            if i_trace==0:
                traces_trials['bins_trial'].extend([bins_trial-bins_trial[0]+trial_offset[0]])
                licksL = binned_statistic(events['Lick L (raw)'], events['Lick L (raw)'], statistic='count', bins=bins_trial, range=[bins_trial.min(), bins_trial.max()])[0]
                licksR = binned_statistic(events['Lick R (raw)'], events['Lick R (raw)'], statistic='count', bins=bins_trial, range=[bins_trial.min(), bins_trial.max()])[0]
                traces_trials['Lick L (raw)'].extend([licksL])
                traces_trials['Lick R (raw)'].extend([licksR])
    df_new = pd.concat([df_new, pd.DataFrame(traces_trials)], axis=1)
    return df_new
