import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# close all:
plt.close('all')

# Image:
image = np.load('./PointTargets-Result.npy')
Image_Modulus = np.abs(image)

# 常数定义:
C = 3e8  # 光速

# chirp 频率调制带宽为 30 MHz:
Br = 30e+6

# 目标区域参数:
Xcenter = 512 - 1
Ycenter = 256 - 1

# 天线参数:
D = 4  # 方位向天线长度，即方位向合成孔径的长度

# 分辨率参数设置:
DX = C / 2 / Br  # 距离向分辨率 ρ_r = C / (2 * B_r)
DY = D / 2  # 方位向分辨率  ρ_a = v_a / B_a，v_a为SAR移动速度，B_a为SAR方位向多普勒带宽。当小斜视角的情况下，ρ_a = D / 2

# # ROI:
# ROI_Center = 630,
# H, W = 20 * DY, 20 * DX

# print(Image_Modulus.max(), Image_Modulus.min())

# Get Mask:
Mask_threshold = 263.5
Mask_Condition = Image_Modulus > Mask_threshold
Mask_Condition = Mask_Condition.astype(int)
y, x = np.where(Mask_Condition == 1)

# Segmentation:
ROI_Mask = image * Mask_Condition
Mod_Mask = np.abs(ROI_Mask)

ROI_Center = y + 1, x + 1
HW = 32
y0, x0 = ROI_Center[0] - HW / 2., ROI_Center[1] - HW / 2.
y0, x0 = int(y0), int(x0)
print('ROI Points:', y0, x0, y0 + HW, x0 + HW)
print('ROI Center:', ROI_Center)

# Get ROI:
ROI = image[y0:y0 + HW, x0:x0 + HW]
print('ROI Shape:', ROI.shape)

# show:
plt.figure(0)
plt.imshow(Image_Modulus, cmap='gray')
plt.title('Modulus of Image')

plt.figure(1)
plt.imshow(Mod_Mask, cmap='gray')
plt.title('Mask with Threshold()')

# BBox:
Image_Modulus_BBox = Image_Modulus.copy()
Image_Modulus_BBox_CV = Image_Modulus_BBox * 255 / Image_Modulus_BBox.max()
Image_Modulus_BBox_CV = Image_Modulus_BBox_CV.astype(np.uint8)

# Gray to 3 Channels:
Image_Modulus_BBox_CV = cv.cvtColor(Image_Modulus_BBox_CV, cv.COLOR_GRAY2BGR)

# Draw:
Image_Modulus_BBox_CV = cv.rectangle(Image_Modulus_BBox_CV.copy(), pt1=(x0, y0), pt2=(x0 + HW, y0 + HW),
                                     color=(255, 0, 0), thickness=2)

plt.figure(2)
plt.imshow(Image_Modulus_BBox_CV, cmap='gray')
plt.title('Image + BBox ')

plt.figure(3)
plt.imshow(np.abs(ROI), cmap='gray')
plt.title('Region of Interest')

np.save('./ROI-215-773-245-803.npy', ROI)

plt.show()

if __name__ == '__main__':
    pass
