#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:07:04 2018

This script plots a set of RWST coefficients

@author: bruno
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

inputDir = "/home/bruno/Desktop/These ENS/Python/Papier-Polarisation/RWSTComputation/RWST/Model0/MHD/9_9_standard/"
inputRWSTFilename = "I_y15_RWST_J7L8S0.npy"

# ScatNet parameters
J = int (inputRWSTFilename.split ('J') [-1][0])
L = int (inputRWSTFilename.split ('L') [-1][0])
oversampling = int (inputRWSTFilename.split ('S') [-1][0])
M = 2

# Load RWST data
data = np.load (os.path.join (inputDir, inputRWSTFilename))

# RWST arrays initialization
chi2reducedS1   = data [0][0][0]
chi2reducedS2   = data [0][1][0]
RWSTMean        = data [1][0]
RWSTMeanErr     = data [1][1]
RWST1Iso        = data [2][0]
RWST1IsoErr     = data [2][1]
RWST1Aniso      = data [3][0]
RWST1AnisoErr   = data [3][1]
RWST2Iso1       = data [4][0]
RWST2Iso1Err    = data [4][1]
RWST2Iso2       = data [5][0]
RWST2Iso2Err    = data [5][1]
RWST2Aniso1     = data [6][0]
RWST2Aniso1Err  = data [6][1]
RWST2Aniso2     = data [7][0]
RWST2Aniso2Err  = data [7][1]

# Theta ref angles
thetaRef1       = data [8][0]
thetaRef1Err    = data [8][1]
thetaRef2       = data [9][0]
thetaRef2Err    = data [9][1]

################ PLOT PARAMS ################

mpl.rc('text', usetex = True)
params = {
  'text.latex.preamble': [r'\usepackage{amsmath}', r'\boldmath'],
  'image.origin': 'lower',
  'image.interpolation': 'nearest',
#  'image.cmap': 'jet',
  'savefig.dpi': 100,
  'axes.labelsize': 16,
  'axes.titlesize': 12,
  'font.size': 12,
  'legend.fontsize': 12,
  'xtick.labelsize': 12,
  'ytick.labelsize': 12,
  'xtick.color': 'black',
  'font.family': 'serif',
  'axes.grid': False,
  'font.weight': 'bold',
  'mathtext.fontset': 'custom',
  'mathtext.default': 'regular',
  }
mpl.rcParams.update(params)

# m = 1 RWST

fig, ax = plt.subplots (2, 2, figsize = (10, 8))
plt.suptitle (inputRWSTFilename.replace ("_", "\_") + " - $m = 1$")
xVals = np.array (range (J))

ax [0, 0].plot (xVals, RWST1Iso)
ax [0, 0].fill_between (xVals, RWST1Iso - RWST1IsoErr, RWST1Iso + RWST1IsoErr, alpha = 0.2)
ax [0, 0].set_xlabel ('$j_1$')
ax [0, 0].set_ylabel ("$\\widehat{S}_1^{\\text{iso}}(j_1)$")

ax [0, 1].plot (xVals, RWST1Aniso)
ax [0, 1].fill_between (xVals, RWST1Aniso - RWST1AnisoErr, RWST1Aniso + RWST1AnisoErr, alpha = 0.2)
ax [0, 1].set_xlabel ('$j_1$')
ax [0, 1].set_ylabel ("$\\widehat{S}_1^{\\text{aniso}}(j_1)$")

ax [1, 0].plot (xVals, thetaRef1)
ax [1, 0].fill_between (xVals, thetaRef1 - thetaRef1Err, thetaRef1 + thetaRef1Err, alpha = 0.2)
ax [1, 0].set_xlabel ('$j_1$')
ax [1, 0].set_ylabel ("$\\theta^{\\text{ref,1}}(j_1)$")

ax [1, 1].plot (xVals, chi2reducedS1)
ax [1, 1].set_xlabel ('$j_1$')
ax [1, 1].set_ylabel ("$\\chi^{2, \\text{S1}}_r(j_1)$")

plt.tight_layout (rect=[0, 0.03, 1, 0.95])

# m = 2 RWST

fig, ax = plt.subplots (3, 2, figsize = (10, 12))
plt.suptitle (inputRWSTFilename.replace ("_", "\_") + " - $m = 2$")
xVals = np.array (range (J))

for i in xVals [:-1]:
    if (i != J - 2):
        ax [0, 0].plot (np.arange (i + 1, J), RWST2Iso1 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
        ax [0, 0].fill_between (np.arange (i + 1, J), RWST2Iso1 [i, i + 1:] - RWST2Iso1Err [i, i + 1:], RWST2Iso1 [i, i + 1:] + RWST2Iso1Err [i, i + 1:], alpha = 0.2)
    else:
        ax [0, 0].errorbar (np.arange (i + 1, J), RWST2Iso1 [i, i + 1:], fmt = '.', yerr = RWST2Iso1Err [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [0, 0].legend ()
ax [0, 0].set_xlabel ('$j_2$')
ax [0, 0].set_ylabel ("$\\widehat{S}_2^{\\text{iso,1}}(j_1, j_2)$")

for i in xVals [:-1]:
    if (i != J - 2):
        ax [0, 1].plot (np.arange (i + 1, J), RWST2Iso2 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
        ax [0, 1].fill_between (np.arange (i + 1, J), RWST2Iso2 [i, i + 1:] - RWST2Iso2Err [i, i + 1:], RWST2Iso2 [i, i + 1:] + RWST2Iso2Err [i, i + 1:], alpha = 0.2)
    else:
        ax [0, 1].errorbar (np.arange (i + 1, J), RWST2Iso2 [i, i + 1:], fmt = '.', yerr = RWST2Iso2Err [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [0, 1].legend ()
ax [0, 1].set_xlabel ('$j_2$')
ax [0, 1].set_ylabel ("$\\widehat{S}_2^{\\text{iso,2}}(j_1, j_2)$")

for i in xVals [:-1]:
    if (i != J - 2):
        ax [1, 0].plot (np.arange (i + 1, J), RWST2Aniso1 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
        ax [1, 0].fill_between (np.arange (i + 1, J), RWST2Aniso1 [i, i + 1:] - RWST2Aniso1Err [i, i + 1:], RWST2Aniso1 [i, i + 1:] + RWST2Aniso1Err [i, i + 1:], alpha = 0.2)
    else:
        ax [1, 0].errorbar (np.arange (i + 1, J), RWST2Aniso1 [i, i + 1:], fmt = '.', yerr = RWST2Aniso1Err [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [1, 0].legend ()
ax [1, 0].set_xlabel ('$j_2$')
ax [1, 0].set_ylabel ("$\\widehat{S}_2^{\\text{aniso,1}}(j_1, j_2)$")

for i in xVals [:-1]:
    if (i != J - 2):
        ax [1, 1].plot (np.arange (i + 1, J), RWST2Aniso2 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
        ax [1, 1].fill_between (np.arange (i + 1, J), RWST2Aniso2 [i, i + 1:] - RWST2Aniso2Err [i, i + 1:], RWST2Aniso2 [i, i + 1:] + RWST2Aniso2Err [i, i + 1:], alpha = 0.2)
    else:
        ax [1, 1].errorbar (np.arange (i + 1, J), RWST2Aniso2 [i, i + 1:], fmt = '.', yerr = RWST2Aniso2Err [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [1, 1].legend ()
ax [1, 1].set_xlabel ('$j_2$')
ax [1, 1].set_ylabel ("$\\widehat{S}_2^{\\text{aniso,2}}(j_1, j_2)$")

for i in xVals [:-1]:
    if (i != J - 2):
        ax [2, 0].plot (np.arange (i + 1, J), thetaRef2 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
        ax [2, 0].fill_between (np.arange (i + 1, J), thetaRef2 [i, i + 1:] - thetaRef2Err [i, i + 1:], thetaRef2 [i, i + 1:] + thetaRef2Err [i, i + 1:], alpha = 0.2)
    else:
        ax [2, 0].errorbar (np.arange (i + 1, J), thetaRef2 [i, i + 1:], fmt = '.', yerr = thetaRef2Err [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [2, 0].legend ()
ax [2, 0].set_xlabel ('$j_2$')
ax [2, 0].set_ylabel ("$\\theta^{\\text{ref,2}}(j_1, j_2)$")

for i in xVals [:-1]:
    ax [2, 1].plot (np.arange (i + 1, J), chi2reducedS2 [i, i + 1:], label = "$j_1 = " + str (i) + "$")
ax [2, 1].legend ()
ax [2, 1].set_xlabel ('$j_2$')
ax [2, 1].set_ylabel ("$\\chi^{2, \\text{S2}}_r(j_1, j_2)$")

plt.tight_layout (rect=[0, 0.03, 1, 0.95])
