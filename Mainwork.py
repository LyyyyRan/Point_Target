import subprocess
import numpy as np

# RCS Setting:
# RCS = 122.39
RCS = 10
RCS_dB = 10 * np.log10(RCS)

print('RCS', RCS)
print('RCS_dB', RCS_dB)

# RDA:
subprocess.run(['python', './RD-Imaging.py', '{}'.format(RCS)], capture_output=False)

# Get ROI:
subprocess.run(['python', './GetROI.py'], capture_output=False)

# Upsampling ROI:
subprocess.run(['python', './UpSampling.py'], capture_output=False)

# Extract Energy and Calibration:
subprocess.run(['python', './EnergyExtract.py', '{}'.format(RCS)], capture_output=False)

if __name__ == '__main__':
    pass
