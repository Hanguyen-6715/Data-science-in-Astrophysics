#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:38:59 2018

This script computes the Reduced Wavelet Scattering Transform coefficients from a set of WST coefficients (see Allys et al., 2019) and stores it in a numpy file.

@author: bruno
"""

import os
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

# Params
inputDir = ""
outputDir = ""
filename = "?"
complexMap = False

print ("Processing " + filename + "...")

# ScatNet parameters
J = int (filename.split ('J') [-1][0])
L = int (filename.split ('L') [-1][0])
Leff = L
if complexMap:
    Leff = 2 * Leff
oversampling = int (filename.split ('S') [-1][0])
M = 2

# Number of WST coefficients per layer
nbM0 = 1
nbM1 = J * Leff
nbM2 = int (Leff ** 2 * J * (J - 1) / 2)

# Meta array
meta = np.zeros ((nbM0 + nbM1 + nbM2, 5))
cntTmp = 1
for j1 in range (J):
    for theta1 in range (1, Leff + 1):
        meta [cntTmp] = [1, j1, theta1, 0, 0]
        cntTmp += 1
for j1 in range (J):
    for theta1 in range (1, Leff + 1):
        for j2 in range (j1 + 1, J):
            for theta2 in range (1, Leff + 1):
                meta [cntTmp] = [2, j1, theta1, j2, theta2]
                cntTmp += 1

# Load Data
PPWSTmean   = np.load (os.path.join (dirFile, filename))
PPWSTcov    = np.load (os.path.join (dirFile, filename.replace ('mean', 'cov')))

# For now we only keep diagonal vavlues to avoid strange fits...
# The estimation of the covariance matrix might be inaccurate
for i in range (PPWSTcov.shape [0]):
    for j in range (PPWSTcov.shape [1]):
        if i != j:
            PPWSTcov [i, j] = 0.0

# RWST arrays initialization
RWSTMean        = np.zeros (1)
RWSTMeanErr     = np.zeros (1)
RWST1Iso        = np.zeros (J)
RWST1IsoErr     = np.zeros (J)
RWST1Aniso      = np.zeros (J)
RWST1AnisoErr   = np.zeros (J)
RWST2Iso1       = np.zeros ((J, J))
RWST2Iso1Err    = np.zeros ((J, J))
RWST2Iso2       = np.zeros ((J, J))
RWST2Iso2Err    = np.zeros ((J, J))
RWST2Aniso1     = np.zeros ((J, J))
RWST2Aniso1Err  = np.zeros ((J, J))
RWST2Aniso2     = np.zeros ((J, J))
RWST2Aniso2Err  = np.zeros ((J, J))

# Theta ref angles
thetaRef1       = np.zeros (J)
thetaRef1Err    = np.zeros (J)
thetaRef2       = np.zeros ((J, J))
thetaRef2Err    = np.zeros ((J, J))

# Chi2
chi2S1          = np.zeros (J)
chi2S2          = np.zeros ((J, J))

############################
##### RWST COMPUTATION #####
############################

# Layer 0 -> nothing to do
RWSTMean    [0] = PPWSTmean [0]
RWSTMeanErr [0] = np.sqrt (PPWSTcov  [0, 0])

###########
# Layer 1 #
###########

# Generate array theta1 values
theta1Arr = np.arange (1, Leff + 1)

# Fit function for m=1 coeffs
def FnRWSTS1 (theta1, a0, a1, a2):
    return a0 + a1 * np.cos (2 * np.pi * (theta1 - a2) / L)

for j1 in range (J):
    # We get the optimized parameters aOpt with their associated covariance matrix that minimize the sum of the normalized squared residuals
    aOpt, aCov = opt.curve_fit (FnRWSTS1, theta1Arr, PPWSTmean [1 + j1 * Leff: 1 + (j1 + 1) * Leff], p0 = np.ones (3), sigma = PPWSTcov [1 + j1 * Leff: 1 + (j1 + 1) * Leff, 1 + j1 * Leff: 1 + (j1 + 1) * Leff])
    
    # Extraction of the parameters and their error
    RWST1Iso       [j1]    = aOpt [0]
    RWST1IsoErr    [j1]    = np.sqrt (aCov [0, 0])
    RWST1Aniso     [j1]    = aOpt [1]
    RWST1AnisoErr  [j1]    = np.sqrt (aCov [1, 1])
    thetaRef1      [j1]    = aOpt [2]
    thetaRef1Err   [j1]    = np.sqrt (aCov [2, 2])
    
    # Chi2 (reduced) computation
    x = PPWSTmean [1 + j1 * Leff: 1 + (j1 + 1) * Leff] - FnRWSTS1 (theta1Arr, RWST1Iso [j1], RWST1Aniso [j1], thetaRef1 [j1])
    chi2S1 [j1] = x.T @ la.inv (PPWSTcov [1 + j1 * Leff:1 + (j1 + 1) * Leff, 1 + j1 * Leff:1 + (j1 + 1) * Leff]) @ x  / (len (x) - 3)
    
###########
# Layer 2 #
###########

# Generate array of pairs of theta1/2 values
theta1Arr = np.zeros (L * Leff) # We need to exclude redundant information (ie theta_2 should stay in [0, \pi[ to avoid null eigenvalues in the covariance matrix])
theta2Arr = np.zeros (L * Leff) # We need to exclude redundant information (ie theta_2 should stay in [0, \pi[ to avoid null eigenvalues in the covariance matrix])
for theta1 in range (Leff):
    for theta2 in range (L):
        theta1Arr [theta1 * L + theta2] = theta1 + 1
        theta2Arr [theta1 * L + theta2] = theta2 + 1
        
# We need to order data
yFullData = np.zeros ((int((J * (J - 1))/ 2), L * Leff)) # We need to exclude redundant information (ie theta_2 should stay in [0, \pi[ to avoid null eigenvalues in the covariance matrix])
yFullDataCov = np.zeros ((int((J * (J - 1))/ 2), L * Leff, L * Leff)) # We need to exclude redundant information (ie theta_2 should stay in [0, \pi[ to avoid null eigenvalues in the covariance matrix])
cntj1j2 = 0
for j1 in range (J):
    for j2 in range (j1 + 1, J):
        filt = np.logical_and (np.logical_and (meta [:, 1] == j1, meta [:, 3] == j2), meta [:, 4] <= L)
        yFullData [cntj1j2] = PPWSTmean [filt]
        yFullDataCov [cntj1j2] = np.reshape (PPWSTcov [np.outer (filt, filt)], (L * Leff, L * Leff))
        cntj1j2 += 1
        
# Fit function for given values of j1/j2/theta2ref
def FnRWSTS2 (theta, a0, a1, a2, a3, a4):
    theta1, theta2 = theta
    return a0 + a1 * np.cos (2 * np.pi * (theta1 - theta2) / L) + a2 * np.cos (2 * np.pi * (theta1 - a4) / L) + a3 * np.cos (2 * np.pi * (theta2 - a4) / L)
 

cntj1j2 = 0
for j1 in range (J):
    for j2 in range (j1 + 1, J):
       
        # We get the optimized parameters aOpt with their associated covariance matrix that minimize the sum of the normalized squared residuals
        aOpt, aCov = opt.curve_fit (FnRWSTS2, (theta1Arr, theta2Arr), yFullData [cntj1j2], p0 = np.ones (5), sigma = yFullDataCov [cntj1j2])
        
        # Extraction of the parameters and their error
        RWST2Iso1       [j1, j2]    = aOpt [0]
        RWST2Iso1Err    [j1, j2]    = np.sqrt (aCov [0, 0])
        RWST2Iso2       [j1, j2]    = aOpt [1]
        RWST2Iso2Err    [j1, j2]    = np.sqrt (aCov [1, 1])
        RWST2Aniso1     [j1, j2]    = aOpt [2]
        RWST2Aniso1Err  [j1, j2]    = np.sqrt (aCov [2, 2])
        RWST2Aniso2     [j1, j2]    = aOpt [3]
        RWST2Aniso2Err  [j1, j2]    = np.sqrt (aCov [3, 3])
        thetaRef2       [j1, j2]    = aOpt [4]
        thetaRef2Err    [j1, j2]    = np.sqrt (aCov [4, 4])
        
        # Chi2 (reduced) computation
        x = yFullData [cntj1j2] - FnRWSTS2 ((theta1Arr, theta2Arr), RWST2Iso1 [j1, j2], RWST2Iso2 [j1, j2], RWST2Aniso1 [j1, j2], RWST2Aniso2 [j1, j2], thetaRef2 [j1, j2])
        chi2S2 [j1, j2] = x.T @ la.inv (yFullDataCov [cntj1j2]) @ x / (len (x) - 5)
        
        cntj1j2 += 1
        
############################
##### RWST FINALIZATION ####
############################

# We need to lift possible degeneracy on some RWST parameters
thetaRef1 = ((thetaRef1 - 1 + L / 2) % L) + 1 - L / 2
thetaRef2 = ((thetaRef2 - 1 + L / 2) % L) + 1 - L / 2
for j in range (J):
    if RWST1Aniso [j] < 0:
        RWST1Aniso [j] = np.abs (RWST1Aniso [j])
        thetaRef1 [j] = thetaRef1 [j] + L / 2
for j in range (1, J):
    if thetaRef1 [j] - thetaRef1 [0] > L / 2:
        thetaRef1 [j] = thetaRef1 [j] - L
    elif thetaRef1 [j] - thetaRef1 [0] < - L / 2:
        thetaRef1 [j] = thetaRef1 [j] + L
for j1 in range (J):
    for j2 in range (j1 + 1, J):
        if RWST2Aniso1 [j1, j2] < 0:
            RWST2Aniso1 [j1, j2] = np.abs (RWST2Aniso1 [j1, j2])
            RWST2Aniso2 [j1, j2] = - RWST2Aniso2 [j1, j2]
            thetaRef2 [j1, j2] = thetaRef2 [j1, j2] + L / 2
for j1 in range (J):
    for j2 in range (j1 + 1, J):
        if thetaRef2 [j1, j2] - thetaRef2 [0, 1] > L / 2:
            thetaRef2 [j1, j2] = thetaRef2 [j1, j2] - L
        elif thetaRef2 [j1, j2] - thetaRef2 [0, 1] < - L / 2:
            thetaRef2 [j1, j2] = thetaRef2 [j1, j2] + L

############################
######## RWST SAVING #######
############################

print ("Fitting process complete. Reduced chi2 values:")
print ("m = 1 :", chi2S1)
print ("m = 2 :", chi2S2)

# RWST save
outputFilename = filename.replace ('WSTmean', 'RWST')
np.save (os.path.join (outputDir, outputFilename), np.array ([[[chi2S1], [chi2S2]], [RWSTMean, RWSTMeanErr], [RWST1Iso, RWST1IsoErr], [RWST1Aniso, RWST1AnisoErr], [RWST2Iso1, RWST2Iso1Err], [RWST2Iso2, RWST2Iso2Err], [RWST2Aniso1, RWST2Aniso1Err], [RWST2Aniso2, RWST2Aniso2Err], [thetaRef1, thetaRef1Err], [thetaRef2, thetaRef2Err]]))

