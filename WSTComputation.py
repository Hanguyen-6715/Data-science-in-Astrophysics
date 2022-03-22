#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example of the WST computation of a FITS image with Kymatio.
WST coefficients are local.
Output is:
- SxNp: 3D numpy array containing the local coefficients (First dimension: coefficient index axis, Second & Third dimension: position index on the image)
- SxIndexNp: 2D numpy array containing a mapping of coefficient index to (m, j_1, theta_1, j_2, theta_2) values (First dimension: (m, j_1, theta_1, j_2, theta_2) variable index, Second dimension: coefficient index)

"""

import os
import torch
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from Scattering2DPar import Scattering2D

################## PARAMETERS ##################

# WST parameters
J = 5
L = 8
OS = 1

# Input filename
filenameFits = "lombardi-column-density-feb18_fix.fits"
outputDir = ""
outputFilenameMean = filenameFits.split('.fits')[0] + "_WSTmean_" + 'J' + str (J) + 'L' + str (L) + 'S' + str (OS) + '.npy'
outputFilenameCov = filenameFits.split('.fits')[0] + "_WSTcov_" + 'J' + str (J) + 'L' + str (L) + 'S' + str (OS) + '.npy'

# Number of coefficients per layer
nbM0 = 1
nbM1 = J * L
nbM2 = L ** 2 * J * (J - 1) // 2

################## LOADING ##################

# Fits file data
hdul = fits.open (filenameFits)
data = hdul [0].data
data = data.byteswap ().newbyteorder () # Swap byte order from big endian (FITS standard) to little endian (demanded by torch tensors)

################## PLOTS ##################

#plt.figure ()
#plt.imshow (data)
#plt.colorbar ()
#plt.show ()

################## WST KYMATIO ##################

# Build a Scattering2D object containing relevant filters
# Add cache=True parameter to save filters in a file and reload it quickly next time
scattering = Scattering2D (data.shape [0], data.shape [1], J, L, OS)

# Build a 4D pytorch tensor from data:
# First dimension is the size of the batch: the length along this dimension is typically the number of input images (one image for this example)
# Second dimension is the number of channels per image: 3 channels for an RGB image for example (only one channel for this example)
# Third & Fourth dimension: dimensions of the image
x = torch.DoubleTensor (np.expand_dims (np.expand_dims (data, axis = 0), axis = 0))
print (x.size ()) # Print the size of the input tensor

# Compute WST coefficients and reorder the coefficients with the order of ScatNet software
Sx, SxIndex = scattering.forward (x)
Sx, SxIndex = scattering.scatnetOrder (Sx, SxIndex)

# Get numpy array from pytorch tensors
SxNp = Sx [0, 0, :, :, :].numpy ()
SxIndexNp = SxIndex.numpy ()

############################ Normalization / Mean-Cov / Save

def Normalization (Smat, J, L):
    logSmat = np.log2 (Smat)
    logSmatNorm = np.log2 (Smat)
    count = 1
    for i in range (0, J):
        for j in range (1, L + 1):
            logSmatNorm [count, :, :] = logSmatNorm [count, :, :] - logSmat [0, :, :]
            count += 1
    for i1 in range (0, J - 1):
        for j1 in range (1, L + 1):
            for i2 in range (i1 + 1, J):
                for j2 in range (1, L + 1):
                    logSmatNorm [count, :, :] = logSmatNorm [count, :, :] - logSmat [1 + i1 * L + (j1 - 1), :, :]
                    count += 1
    return logSmatNorm

# Normalization
logSmatNorm = Normalization (SxNp, J, L)
                    
# Mean computation
WSTmean = np.mean (logSmatNorm, axis = (1, 2))

# Covariance matrix computation
logSmatNormTmp = np.reshape (logSmatNorm, (logSmatNorm.shape [0], logSmatNorm.shape [1] * logSmatNorm.shape [2]))
WSTcov = np.cov (logSmatNormTmp, bias = True) / (logSmatNorm.shape [1] * logSmatNorm.shape [2])

# Save final mean/std in a npy file
np.save (os.path.join (outputDir, outputFilenameMean), WSTmean)
np.save (os.path.join (outputDir, outputFilenameCov), WSTcov)
