import numpy        as np
from scipy.signal   import butter, filtfilt
from scipy          import ndimage

import matplotlib.pyplot as plt
class ButterworthFilters:
    def __init__(self, fs):
        self.fs = fs

    def butter_highpass(self, cutoff, order=1):
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_lowpass(self, cutoff, order=1):
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def highpass_filter(self, data, cutoff=0.5, order=1):
        b, a = self.butter_highpass(cutoff, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def lowpass_filter(self, data, cutoff=10, order=1):
        b, a = self.butter_lowpass(cutoff, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    

class PPGdenoising:
    def __init__(self, PPG_signal, hf_noise_window, lf_noise_window):
        self.hfw        = hf_noise_window
        self.lfw        = lf_noise_window
        self.signal     = PPG_signal

    def normalized(self):
        PPG1 = np.zeros(len(self.signal))
        max_val = np.max(self.signal)
        min_val = np.min(self.signal)
        PPG1    = (self.signal - min_val) / (max_val - min_val)
        return PPG1
    
    def PPG_median_filter(self):

        PPG1 = self.normalized()
        PPG3 = ndimage.median_filter(PPG1, size=self.hfw, mode='constant')
    
        PPG4 = ndimage.median_filter(PPG3, size=self.lfw,mode='constant')
      
        PPG5 = PPG3 - PPG4
      
        return [PPG3,PPG4,PPG5]
