import numpy as np
import matplotlib.pyplot as plt
from np2mtlb import FFT_Azimuth, FFT_Range, FFTShift, IFFT_Azimuth, IFFT_Range
from utils import UpSampling, ZeroPadding

# close all:
plt.close('all')

# Read ROI from path:
ROI = np.load('./ROI-215-773-245-803.npy')
ROI_f = FFT_Azimuth(FFT_Range(ROI, shift=True), shift=True)  # Get Frequency Map
# ROI_f = np.fft.fft2(ROI)

# Get Modulus:
ROI_Modulus = np.abs(ROI)
ROI_f_Modulus = np.abs(ROI_f)

# imshow:
plt.figure('Region of Interest')
plt.imshow(ROI_Modulus, cmap='gray')
plt.title('Region of Interest')

plt.figure('Frequency Map')
plt.imshow(ROI_f_Modulus, cmap='gray')
plt.title('Frequency Map')

# Zero Padding in Frequency Domain:
ROI_f_Upsampling = ZeroPadding(ROI_f)
ROI_f_Upsampling_Mod = np.abs(ROI_f_Upsampling)

# Back to Range-Azimuth Space Domain:
ROI_Upsampling = IFFT_Range(IFFT_Azimuth(ROI_f_Upsampling, shift=True), shift=True)
ROI_Upsampling_Modulus = np.abs(ROI_Upsampling)

plt.figure('Frequency Map Upsampling')
plt.imshow(ROI_f_Upsampling_Mod, cmap='gray')
plt.title('Frequency Map Upsampling')

plt.figure('ROI Upsampling')
plt.imshow(ROI_Upsampling_Modulus, cmap='gray')
plt.title('ROI Upsampling')

print('Old Shape:', ROI.shape)
print('New Shape:', ROI_Upsampling.shape)

# save:
np.save('./ROI-Upsampling.npy', ROI_Upsampling)

# show all:
plt.show()
