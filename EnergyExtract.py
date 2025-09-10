import numpy as np
import matplotlib.pyplot as plt

# read ROI:
ROI = np.load('./ROI-Upsampling.npy')
ROI_Mod = np.abs(ROI)

# Show image:
plt.figure('Region of Interest')
plt.imshow(ROI_Mod, cmap='gray')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Region of Interest"')

# Get Peak Point:
peak_y, peak_x = np.where(ROI_Mod == ROI_Mod.max())

# Get shape of ROI:
Na, Nr = ROI_Mod.shape
Na, Nr = int(Na), int(Nr)

#

# Show all:
plt.show()
