"""FIP Preprocessing Library """
import os
import csv
import numpy as  np
import pylab as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import glob
# ---------------------------------------------------------------------------------------

# removing first few seconds
def tc_crop(tc, nFrame2cut):
    tc_cropped = tc[nFrame2cut:]
    return tc_cropped

# Median filtering to remove electrical artifact.
def tc_medfilt(tc, kernelSize):
    tc_filtered = medfilt(tc, kernel_size=kernelSize)
    return tc_filtered

# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
def tc_lowcut(tc, sampling_rate):
    b,a = butter(2, 9, btype='low', fs=sampling_rate)
    tc_filtered = filtfilt(b,a, tc)
    return tc_filtered

# fit with polynomial to remove bleaching artifact 
def tc_polyfit(tc, sampling_rate, degree):
    time_seconds = np.arange(len(tc)) /sampling_rate 
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly, coefs

# setting up sliding baseline to calculate dF/F
def tc_slidingbase(tc, sampling_rate):
    b,a = butter(2, 0.0001, btype='low', fs=sampling_rate)
    tc_base = filtfilt(b,a, tc, padtype='even')
    return tc_base

# obtain dF/F using median of values within sliding baseline 
def tc_dFF(tc, tc_base, b_percentile):
    tc_dFoF = tc/tc_base
    sort = np.sort(tc_dFoF)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    tc_dFoF = tc_dFoF - b_median
    return tc_dFoF

# fill in the gap left by cropping out the first few timesteps
def tc_filling(tc, nFrame2cut):
    tc_filled = np.append(np.ones([nFrame2cut,1])*tc[0], tc)
    return tc_filled
    
# bi-exponential fit
def func(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)
 
def tc_expfit(tc, sampling_rate=20):
    time_seconds = np.arange(len(tc))/sampling_rate
    popt, pcov = curve_fit(func,time_seconds,tc)
    tc_exp = func(time_seconds, popt[0], popt[1], popt[2], popt[3])
    return tc_exp, popt

# Preprocessing total function
def tc_preprocess(tc, method = 'poly', nFrame2cut=100, kernelSize=1, sampling_rate=20, degree=4, b_percentile=0.7):
    # Standard parameters values
        # nFibers = 2
        # nColor = 3
        # sampling_rate = 20 #individual channel (not total)
        # nFrame2cut = 100  #crop initial n frames
        # b_percentile = 0.70 #To calculare F0, median of bottom x%
        # BiExpFitIni = [1,1e-3,1,1e-3,1]  #currently not used (instead fitted with 4th-polynomial)
        # kernelSize = 1 #median filter
        # degree = 4 #polyfit
    tc_cropped = tc_crop(tc, nFrame2cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernelSize)
    tc_filtered = tc_lowcut(tc_filtered, sampling_rate)
    
    if method == 'poly':
        tc_fit, tc_coefs = tc_polyfit(tc_filtered, sampling_rate, degree)
    if method == 'exp':
        tc_fit, tc_coefs = tc_expfit(tc_filtered, sampling_rate)        

    tc_estim = tc_filtered - tc_fit # 
    tc_base = tc_slidingbase(tc_filtered, sampling_rate)
    #tc_dFoF = tc_dFF(tc_filtered, tc_base, b_percentile)
    tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)    
    tc_dFoF = tc_filling(tc_dFoF, nFrame2cut)    
    tc_params = {i_coef:tc_coefs[i_coef] for i_coef in range(len(tc_coefs))}
    tc_qualitymetrics = {'QC_metric':np.nan}    
    tc_params.update(tc_qualitymetrics)
    return tc_dFoF, tc_params

