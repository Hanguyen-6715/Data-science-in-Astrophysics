#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:15:04 2018

@author: bruno
"""

import numpy as np
import matplotlib.pyplot as plt

inputDir = "home/hanguyen/Documents/Archive(1)/"
filenameMean = "lombardi-column-density-feb18_fix_WSTmean_J5L8S1.npy"
filenameCov = "lombardi-column-density-feb18_fix_WSTcov_J5L8S1.npy"
#filenameCov = filenameMean.replace ('mean', 'cov')

# ScatNet parameters
J = int (filenameMean.split ('J') [-1][0])
L = int (filenameMean.split ('L') [-1][0])
if filenameMean.find ('QiU') >= 0:
    L = 2 * L
oversampling = int (filenameMean.split ('S') [-1][0])
M = 2

# Number of WST coefficients per layer
nbM0 = 1
nbM1 = J * L
nbM2 = int (L ** 2 * J * (J - 1) / 2)

# Plot parameters
plotFormat = '-'

# Meta array and plot design preparation
# For each coefficient we save the corresponding vector [m, j_1, theta_1, j_2, theta_2] in meta array
xticks = []
xticksMinor = []
xticklabels = []
xticklabelsMinor = []
tickAlignment = []
meta = np.zeros ((nbM0 + nbM1 + nbM2, 5))
count = 1
for i in range (0, J):
    countTmp = count
    for j in range (1, L + 1):
        meta [count] = [1, i, j, 0, 0]
        count += 1
    xticks.append (int ((count + countTmp)/2))
    xticklabels.append ("j1 = " + str (i))
    tickAlignment.append (-0.1)
for i1 in range (0, J - 1):
    countTmp = count
    for j1 in range (1, L + 1):
        for i2 in range (i1 + 1, J):
            countTmp2 = count
            for j2 in range (1, L + 1):
                meta [count] = [2, i1, j1, i2, j2]
                count += 1
            xticksMinor.append (int ((count + countTmp2)/2))
            xticklabelsMinor.append ("j2 = " + str (i2))
            tickAlignment.append (-0.1)
    xticks.append (int ((count + countTmp)/2))
    xticklabels.append ("j1 = " + str (i1))
    tickAlignment.append (0.0)
xValues = np.array (range (nbM0 + nbM1 + nbM2))

def PlotWithLayout (axis, x, y, ylabel, legend = "", err = None, j1Ticks = True):
    # Plot
    axis.plot (x, y,  plotFormat,  markersize = 2, label = legend)
    if err is not None:
        axis.fill_between (x, y - err, y + err, alpha = 0.2)
    
    # Ticks and grid parameters
    axis.set_ylabel (ylabel)
    if j1Ticks:
        axis.set_xticks (xticks)
        axis.set_xticklabels (xticklabels, rotation = 'vertical')
    else:
        axis.set_xticks ([])
        axis.set_xticklabels ([], visible = False)
    axis.set_xticks (xticksMinor, minor = True)
    axis.set_xticklabels (xticklabelsMinor, rotation = 'vertical', minor = True)
    for t, y in zip (axis.get_xticklabels (), tickAlignment):
        t.set_y (y)
    axis.grid ('off', axis = 'x')
    axis.grid ('off', axis = 'x', which = 'minor')
    axis.tick_params (axis = 'x', which = 'minor', direction = 'out', length = 5)
    axis.tick_params (axis = 'x', which = 'major', bottom = False, top = False)
    axis.set_xlim (x.min () - 1, x.max () + 1)
    axis.margins (0.2)
    
    # Plot separators
    count = 1
    for i in range (0, J):
        if count in x:
            axis.axvline (count, color = 'black', alpha = 0.5, linestyle = 'dashed')
        for j in range (1, L + 1):
            count += 1
    for i1 in range (0, J - 1):
        if count in x:
            axis.axvline (count, color = 'black', alpha = 0.5, linestyle = 'dashed')
        for j1 in range (1, L + 1):
            if j1 > 1:
                if count in x:
                    axis.axvline (count, color = 'black', alpha = 0.2, linestyle = 'dashed')
            for i2 in range (i1 + 1, J):
                if i2 > i1 + 1:
                    if count in x:
                        axis.axvline (count, color = 'black', alpha = 0.1, linestyle = ':')
                for j2 in range (1, L + 1):
                    count += 1
    if count - 1 in x:
        axis.axvline (count, color = 'black', alpha = 0.5, linestyle = 'dashed')
        
    if legend != "":
        axis.legend ()
        
## Load Data
PPWSTmean   = np.load ("lombardi-column-density-feb18_fix_WSTmean_J5L8S1.npy")
PPWSTcov    = np.load ("lombardi-column-density-feb18_fix_WSTcov_J5L8S1.npy")
PPWSTstd    = np.sqrt (np.diag (PPWSTcov))

# Log2 Plot
PPWSTstd = PPWSTstd / (np.log (2) * PPWSTmean)
PPWSTmean = np.log2 (PPWSTmean)

## Plot WST
fig = plt.figure ()
ax = fig.add_subplot (1,1,1)
xSelection = np.array (range (nbM0, nbM0 + nbM1))
PlotWithLayout (ax, xValues [xSelection], PPWSTmean [xSelection], '$\log_2(\widebar{S}_1)$', legend = "Data", err = PPWSTstd [xSelection])
plt.subplots_adjust (bottom = 0.2)
plt.suptitle (filenameMean + " - m = 1")
plt.show ()
fig = plt.figure (figsize = (30, 5))
ax = fig.add_subplot (1,1,1)
xSelection = np.array (range (nbM0 + nbM1, nbM0 + nbM1 + nbM2))
PlotWithLayout (ax, xValues [xSelection], PPWSTmean [xSelection], '$\log_2(\widebar{S}_2)$', legend = "Data", err = PPWSTstd [xSelection])
plt.subplots_adjust (bottom = 0.2)
plt.suptitle (filenameMean + " - m = 2")
plt.tight_layout (rect=[0, 0.03, 1, 0.95])
plt.show ()
