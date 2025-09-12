import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Light Speed:
C = 3e+8

# Params:
Br = 30e+6  # Br of the chirp signal
D = 4  # Length of Antenna
D_Range = C / 2 / Br  # delta_R = C / (2 * B_r)
D_Azimuth = D / 2  # delta_A = D / 2

# read ROI:
ROI = np.load('./ROI-Upsampling.npy')
ROI_Mod = np.abs(ROI)

# Get Peak Point:
peak_y, peak_x = np.where(ROI_Mod == ROI_Mod.max())

# Get shape of ROI:
Na, Nr = ROI_Mod.shape
Na, Nr = int(Na), int(Nr)

########### integration B ###########
SideLength_B = 30
NumB = 4 * (SideLength_B ** 2)

# [a, b) is used in numpy index:
ROI_B1 = ROI_Mod[ : SideLength_B,  : SideLength_B]
ROI_B2 = ROI_Mod[ : SideLength_B, Nr - SideLength_B : ]
ROI_B3 = ROI_Mod[Na - SideLength_B : ,  : SideLength_B]
ROI_B4 = ROI_Mod[Na - SideLength_B : , Nr - SideLength_B : ]

# DN_B ** 2:
IntegrationB = (ROI_B1 ** 2).sum() + (ROI_B2 ** 2).sum() + (ROI_B3 ** 2).sum() + (ROI_B4 ** 2).sum()

########### integration A ###########
A_Mask = np.zeros_like(ROI_Mod, dtype=int)  # [a, b)
A_Mask[:, 38 : 54 + 1] = 1
A_Mask[42 : 52 + 1, :] = 1
NumA = A_Mask.sum()

# Segmentation:
ROI_A = ROI_Mod * A_Mask

# DN_A ** 2:
IntegrationA = (ROI_A ** 2).sum()

########### Calculate Energy ###########
# Num_UpSampling =
# Energy = (IntegrationA - (NumA * IntegrationB / NumB)) * 0.4 * 0.2 * D_Azimuth * D_Range  # Unknow delta_a and delta_r, so that use part of their Spatial Resolution
Energy = (IntegrationA - (NumA * IntegrationB / NumB)) * 1.7354013507078536 * 3.2261778224237325
# Energy /= 9.  # cuz by Upsampling, whether need to perform?
Energy_dB = 10 * np.log10(Energy)
print('Extracted Energy:', Energy)
print('Extracted Energy/dB:', Energy_dB)
print()

# RCS:
# RCS = 122.39
RCS = 25.136
RCS_dB = 10 * np.log10(RCS)

print('Setting RCS:', RCS)
print('Setting RCS/dB:', RCS_dB)
print()

# Calibration:
Calibration_K = Energy / RCS
Calibration_K_dB = 10 * np.log10(Calibration_K)

print('Calibration Constant K:', Calibration_K)
print('Calibration Constant K/dB:', Calibration_K_dB)

# To show:
ROI_BBox_show = ROI_Mod.copy()  # Copy
ROI_BBox_show = ROI_BBox_show / ROI_BBox_show.max()  # squeeze val domain
ROI_BBox_show = ROI_BBox_show * 255  # put val domain to be 0~255
ROI_BBox_show = np.uint8(ROI_BBox_show)
ROI_BBox_show = cv.cvtColor(ROI_BBox_show, cv.COLOR_GRAY2BGR)  # 1-Channel to 3-Channels

# Drew B BBoxs: (Range, Azimuth) is used in OpenCV
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 0), (SideLength_B - 1, SideLength_B - 1), color=(0, 0, 255), thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, 0), (Nr - 1, SideLength_B - 1), color=(0, 0, 255), thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, Na - SideLength_B), (SideLength_B - 1, Na - 1), color=(0, 0, 255), thickness=1)
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, Na - SideLength_B), (Nr, Na - 1), color=(0, 0, 255), thickness=1)

# Drew A BBoxs: (Range, Azimuth) is used in OpenCV
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (38, 0), (54, Na - 1), color=(255, 0, 0), thickness=1)  # Azimuth longer
ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 42), (Nr - 1, 52), color=(255, 0, 0), thickness=1)  # Range longer

# Show image:
plt.figure('Region of Interest')
plt.imshow(ROI_BBox_show)
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Region of Interest"')

# Show Mask_A
plt.figure('Mask A')
plt.imshow(A_Mask, cmap='gray')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('"Mask A"')

# Show all:
plt.show()
