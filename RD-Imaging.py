import numpy as np
from np2mtlb import nextpow2, FFT_Range, FFT_Azimuth, FFTShift, apostrophe, pointwise_apostrophe, IFFT_Range, \
    IFFT_Azimuth
import matplotlib.pyplot as plt

# 常数定义
C = 3e8  # 光速

# 雷达参数
Fc = 1e+9  # 载频1GHz
lamda = C / Fc  # 波长λ

# 目标区域参数
Xmin = 0  # 目标区域方位向范围[Xmin,Xmax]
Xmax = 50
Yc = 10000  # 成像区域中线
Y0 = 500  # 目标区域距离向范围[Yc-Y0, Yc+Y0], 成像宽度为2*Y0

# 轨道参数
V = 100  # SAR的运动速度 100 m/s
H = 5000  # 高度 5000 m，机载SAR。星载SAR一般在低轨（500km ~ 2000km）
R0 = np.sqrt(Yc ** 2 + H ** 2)  # 最短距离

# 天线参数
D = 4  # 方位向天线长度，即方位向合成孔径的长度
Lsar = lamda * R0 / D  # SAR合成孔径长度，即方位向合成孔径的等效天线长度。《合成孔径雷达成像——算法与实现》P.100
Tsar = Lsar / V  # SAR照射时间

# 慢时间域参数
Ka = -2 * V ** 2 / lamda / R0  # 多普勒频域调频率P.93
Ba = np.abs(Ka * Tsar)  # 多普勒频率调制带宽
PRF = Ba  # 脉冲重复频率，PRF其实为多普勒频率的采样率，又为复频率，所以等于Ba.P.93
PRT = 1 / PRF  # 脉冲重复时间
ds = PRT  # 慢时域的时间步长，单位是时间单位
Nslow = np.ceil((Xmax - Xmin + Lsar) / V / ds)  # 慢时域的采样数，ceil为取整函数，结合P.76的图理解
Nslow = 2 ** nextpow2(Nslow)  # nextpow2 为最靠近2的幂次函数，这里为fft变换做准备

# 慢时间域的时间矩阵:
sn = np.linspace(start=(Xmin - Lsar / 2) / V, stop=(Xmax + Lsar / 2) / V, num=Nslow)
sn = np.expand_dims(sn, 0)

PRT = (Xmax - Xmin + Lsar) / V / Nslow  # 由于 Nslow 改变了，所以相应的一些参数也需要更新，周期减小了
PRF = 1 / PRT
ds = PRT

# 快时间域参数设置
Tr = 5e-6  # 脉冲持续时间5us
Br = 30e+6  # chirp频率调制带宽为30MHz
Kr = Br / Tr  # chirp调频率
Fsr = 2 * Br  # 快时域采样频率，为3倍的带宽
dt = 1 / Fsr  # 快时域采样间隔
Rmin = np.sqrt((Yc - Y0) ** 2 + H ** 2)
Rmax = np.sqrt((Yc + Y0) ** 2 + H ** 2 + (Lsar / 2) ** 2)
Nfast = np.ceil(2 * (Rmax - Rmin) / C / dt + Tr / dt)  # 快时域的采样数量
Nfast = 2 ** nextpow2(Nfast)  # 更新为2的幂次，方便进行fft变换

# 快时域的离散时间矩阵:
tm = np.linspace(2 * Rmin / C, 2 * Rmax / C + Tr, num=Nfast)
tm = np.expand_dims(tm, 0)

dt = (2 * Rmax / C + Tr - 2 * Rmin / C) / Nfast  # 更新间隔
Fsr = 1 / dt

# 分辨率参数设置
DY = C / 2 / Br  # 距离向分辨率 ρ_r = C / (2 * B_r)
DX = D / 2  # 方位向分辨率  ρ_a = v_a / B_a，v_a为SAR移动速度，B_a为SAR方位向多普勒带宽。当小斜视角的情况下，ρ_a = D / 2

# 点目标参数设置
Ntarget = 5  # 点目标的数量

# 只 plot 该方位向坐标的 1 维 距离向信号:
print(Xmin)
view_azimuth = 230
print('view_azimuth', view_azimuth)

# 点目标格式[x,y,反射系数sigma]
# sigma = 122.39
sigma = 25.136
Ptarget = np.array([[Xmin, Yc - 50 * DY, sigma],  # 点目标位置，这里设置了5个点目标，构成一个矩形以及矩形的中心
                    [Xmin + 50 * DX, Yc - 50 * DY, sigma],
                    [Xmin + 25 * DX, Yc, sigma],
                    [Xmin, Yc + 50 * DY, sigma],
                    [Xmin + 50 * DX, Yc + 50 * DY, sigma]])

# 参数显示:
print('Parameters:')
print('Sampling Rate in fast-time domain')
print(Fsr / Br)
print('Sampling Number in fast-time domain')
print(Nfast)
print('Sampling Rate in slow-time domain')
print(PRF / Ba)
print('Sampling Number in slow-time domain')
print(Nslow)
print('Range Resolution')
print(DY)
print('Cross-range Resolution')
print(DX)
print('SAR integration length')
print(Lsar)
print('Position of targets')
print(Ptarget)

# ================================================================
# 生成回波信号
K = Ntarget  # 目标数目
N = Nslow  # 慢时域的采样数，→方位向
M = Nfast  # 快时域的采样数，→距离向
T = Ptarget  # 目标矩阵
Srnm = np.zeros([N, M])  # 生成零矩阵存储回波信号

for k in range(1, K + 1):  # k=1:K  # 总共K个目标，已设置为5个
    sigma = T[k - 1, 3 - 1]  # 得到目标的反射系数
    Dslow = sn * V - T[k - 1, 1 - 1]  # 方位向距离，投影到方位向的距离，即(x - x_t)^2，即[v(s - s_0)]^2，此处为 (s_0 * v - x)^2，sn变量存储各个s_0时刻
    R = np.sqrt(Dslow ** 2 + T[k - 1, 2 - 1] ** 2 + H ** 2)  # 实际距离矩阵
    tau = 2 * R / C  # 回波相对于发射波的延时

    # (t-tau)，其实就是时间矩阵，ones(N,1) 和 ones(1,M) 都是为了将其扩展为矩阵
    tm_expand = np.dot(np.ones([N, 1]), tm)
    tau_expand = np.dot(apostrophe(tau), np.ones([1, M]))
    Dfast = tm_expand - tau_expand

    # 相位，公式参见P.96:
    R_expand = np.dot(apostrophe(R), np.ones([1, M]))
    phase = np.pi * Kr * Dfast ** 2 - (4 * np.pi / lamda) * R_expand

    condition1_1 = 0 < Dfast
    condition1_1 = condition1_1.astype(int)
    condition1_2 = Dfast < Tr
    condition1_2 = condition1_2.astype(int)
    condition1 = condition1_1 * condition1_2  # &&

    condition2 = np.abs(Dslow) < Lsar / 2
    condition2 = condition2.astype(int)
    condition2_expand = np.dot(apostrophe(condition2), np.ones([1, M]))

    Srnm = Srnm + sigma * np.exp(1j * phase) * condition1 * condition2_expand  # 由于是多个目标反射的回波，所以此处进行叠加

# plt.figure('Echo Mod, :]')

plt.figure('Echo Modulus[idx, :]')
plt.plot(np.abs(Srnm[view_azimuth, :]))
plt.xlabel('Range')
plt.ylabel('Modulus')
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.title('Echo Modulus[idx, :]')

plt.figure('Echo image')
plt.imshow(np.abs(Srnm))
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.yticks(ticks=[0, 64, 128, 192, 256, 320, 384, 448, 512],
           labels=[f"{i}" for i in np.around(np.linspace(Xmin - Lsar / 2, Xmax + Lsar / 2, 9)).astype(int)])
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('Echo image')

# ================================================================
# 距离-多普勒算法开始

# 距离向压缩
tr = tm - (2 * Rmin / C)

condition1 = 0 < tr
condition2 = tr < Tr
condition1 = condition1.astype(int)
condition2 = condition2.astype(int)
condition = condition1 * condition2

h_r = np.exp(1j * np.pi * Kr * tr ** 2) * condition
H_r = FFT_Range(h_r, shift=True).conj()
H_r = np.dot(np.ones([N, 1]), H_r)  # expand

Sr = FFT_Range(Srnm, shift=True) * H_r
Sr = IFFT_Range(Sr, shift=True)
Img_AfterRangePulseCompression = np.abs(Sr)

plt.figure('Modulus After Pulse Compression over Range')
plt.plot(Img_AfterRangePulseCompression[view_azimuth, :])
plt.xlabel('Range')
plt.ylabel('Modulus')
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.title('Modulus After Pulse Compression over Range')

plt.figure('After Pulse Compression over Range')
plt.imshow(Img_AfterRangePulseCompression)  # 距离向压缩，未校正距离徙动的图像
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.yticks(ticks=[0, 64, 128, 192, 256, 320, 384, 448, 512],
           labels=[f"{i}" for i in np.around(np.linspace(Xmin - Lsar / 2, Xmax + Lsar / 2, 9)).astype(int)])
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('After Pulse Compression over Range')

# 开始进行距离弯曲补偿正侧视没有距离走动项 主要是因为斜距的变化引起回波包络的徙动
# 补偿方法：最近邻域插值法，具体为：先变换到距离多普勒域，分别对单个像素点计算出距离徙动量，得到距离徙动量与距离分辨率的比值，
# 该比值可能为小数，按照四舍五入的方法近似为整数，而后在该像素点上减去徙动量

# 方位向做fft处理 再在频域做距离弯曲补偿
Sa_RD = FFT_Azimuth(Sr, shift=True)  # 方位向FFT 变为距离多普域进行距离弯曲校正

# 距离徙动运算,由于是正侧视，fdc=0, 只需要进行距离弯曲补偿

# 首先计算距离迁移量矩阵
for n in range(1, N + 1):  # n=1:N  # 总共有N个方位采样
    for m in range(1, M + 1):  # m=1:M  # 每个方位采样上有M个距离采样
        # 距离迁移量P.160；(R0+(m-M/2)*C/2/Fsr)：每个距离向点m的R0更新；
        # (n-N/2)*PRF/N：不同方位向的多普勒频率不一样
        delta_R = (1 / 8) * (lamda / V) ** 2 * (R0 + (m - M / 2) * C / (2 * Fsr)) * ((n - N / 2) * PRF / N) ** 2

        # 此处为delta_R/DY, 距离徒动了几个距离单元:
        RMC = 2 * delta_R * Fsr / C

        # 分解为 RCM: Integers + decimals:
        Integer_RMC = int(np.around(RMC))  # 距离徒动量的整数部分
        decimal_RMC = RMC - Integer_RMC  # 距离徒动量的小数部分

        # index, from zero on:
        n_index, m_index = n - 1, m - 1

        if m + Integer_RMC > M:  # 判断是否超出边界
            Sa_RD[n_index, m_index] = Sa_RD[n_index, int(np.around(M / 2)) - 1]
        else:
            if decimal_RMC >= 0.5:  # 五入
                Sa_RD[n_index, m_index] = Sa_RD[n_index, m_index + Integer_RMC + 1]
            else:  # 四舍
                Sa_RD[n_index, m_index] = Sa_RD[n_index, m_index + Integer_RMC]

# ========================
Img_AfterRCMC = IFFT_Azimuth(Sa_RD, shift=True)  # 距离徙动校正后还原到时域

plt.figure('Modulus After RCMC')
plt.plot(np.abs(Img_AfterRCMC[view_azimuth, :]))
plt.xlabel('Range')
plt.ylabel('Modulus')
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.title('Modulus After RCMC')

plt.figure('After RCMC')
plt.imshow(np.abs(Img_AfterRCMC))  # 距离向压缩，未校正距离徙动的图像
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.yticks(ticks=[0, 64, 128, 192, 256, 320, 384, 448, 512],
           labels=[f"{i}" for i in np.around(np.linspace(Xmin - Lsar / 2, Xmax + Lsar / 2, 9)).astype(int)])
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('After RCMC')

# 方位向压缩:
ta = sn - Xmin / V
condition = np.abs(ta) < Tsar / 2
condition = condition.astype(int)

h_a = np.exp(1j * np.pi * Ka * ta ** 2) * condition
H_a = FFT_Azimuth(h_a, shift=True).conj()  # H( ) == S( )'
H_a = np.dot(pointwise_apostrophe(H_a), np.ones([1, M]))  # expand shape
Img_AfterAzimuthPulseCompression = IFFT_Azimuth(Img_AfterRCMC * H_a, shift=True)

plt.figure('Modulus Pulse Compress over Azimuth')
plt.plot(np.abs(Img_AfterAzimuthPulseCompression[view_azimuth, :]))
plt.xlabel('Range')
plt.ylabel('Modulus')
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.title('Modulus Pulse Compress over Azimuth')

fig, ax = plt.subplots()
plt.imshow(np.abs(Img_AfterAzimuthPulseCompression))
plt.xticks(ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
           labels=[f"{i}" for i in np.around(np.linspace(Yc - Y0, Yc + Y0, 9)).astype(int)])
plt.yticks(ticks=[0, 64, 128, 192, 256, 320, 384, 448, 512],
           labels=[f"{i}" for i in np.around(np.linspace(Xmin - Lsar / 2, Xmax + Lsar / 2, 9)).astype(int)])
ax.axis('scaled')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('After Pulse Compress over Azimuth')

# Get Pixel Size:
Na, Nr = Img_AfterAzimuthPulseCompression.shape
delta_R = (tr[0, -1] - tr[0, 0]) * C / Nr
delta_A = (ta[0, -1] - ta[0, 0]) * V / Na

print('delta_R: {}, delta_A, {}'.format(delta_R, delta_A))

# Get Peak arg and val:
peak_val_arg = np.abs(Img_AfterAzimuthPulseCompression[view_azimuth, :]).argmax()
second_peak_val_arg = np.abs(Img_AfterAzimuthPulseCompression[view_azimuth, :int((10200-9500) * 1024 / 1000)]).argmax()
peak_val = np.abs(Img_AfterAzimuthPulseCompression[view_azimuth, peak_val_arg])
second_peak_val = np.abs(Img_AfterAzimuthPulseCompression[view_azimuth, second_peak_val_arg])

print('peakVal_Arg: {}, second_peakVal_Arg: {}'.format(int(9500 + peak_val_arg * 1000 / 1024), int(9500 + second_peak_val_arg * 1000 / 1024)))
print('peak_val: {}, second_peak_val: {}'.format(peak_val, second_peak_val))

# Get Peak Side Lobe Ratio (PSLR):
PSLR = 10 * np.log10(second_peak_val / peak_val)
whether_satisfy = PSLR < -13
print('PSLR(dB): {}'.format(PSLR), '< -13 dB' if whether_satisfy else '> -13 dB')

# show all:
plt.show()

# save result:
np.save('./PointTargets-Result.npy', Img_AfterAzimuthPulseCompression)
