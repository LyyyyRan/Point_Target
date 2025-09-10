import numpy as np


def ZeroPadding(signal):
    Na, Nr = signal.shape
    new_Na, new_Nr = int(2 ** np.ceil(np.log2(Na))), int(2 ** np.ceil(np.log2(Nr)))
    new_signal = np.pad(signal, (new_Na, new_Nr))
    return new_signal

def UpSampling(signal):
    Na, Nr = signal.shape
    new_Na, new_Nr = int(2 ** np.ceil(np.log2(Na))), int(2 ** np.ceil(np.log2(Nr)))
    new_signal = np.kron(signal, np.ones((new_Na, new_Nr)))
    return new_signal