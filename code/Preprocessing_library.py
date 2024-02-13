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
    return tc_poly, tc_polycoefs

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
    
    
#Preprocessing total function
def tc_preprocess(tc, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile):
    tc_cropped = tc_crop(tc, nFrame2cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernelSize)
    tc_filtered = tc_lowcut(tc_filtered, sampling_rate)
    tc_poly, tc_polycoefs = tc_polyfit(tc_filtered, sampling_rate, degree)
    tc_estim = tc_filtered - tc_poly # 
    tc_base = tc_slidingbase(tc_filtered, sampling_rate)
    #tc_dFoF = tc_dFF(tc_filtered, tc_base, b_percentile)
    tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)    
    tc_dFoF = tc_filling(tc_dFoF, nFrame2cut)
    tc_params = tc_polycoefs
    tc_qualitymetrics = {'NaN':np.nan}
    tc_params.update(tc_qualitymetrics)
    return tc_dFoF, tc_params

