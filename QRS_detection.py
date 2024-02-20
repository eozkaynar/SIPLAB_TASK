import  numpy as np 
import  sys

from    scipy import signal

class Find_R_peaks:
    def __init__(self, ecg_signal, mwi_signal, bp_signal, fs):
        # Initialize parameters
        self.possible_peaks = []
        self.signal_peaks   = []
        self.r_peaks        = []
        self.RR_1           = []  # the eight most-recent beats
        self.RR_2           = []

        self.SLI            = 0  # Integrated signal level
        self.SLF            = 0  # Filtered signal level
        self.NLI            = 0  # Integrated noise level
        self.NLF            = 0  # Filtered noise level
        self.Threshold_I1   = 0  # Integrated threshold 1
        self.Threshold_I2   = 0  # Integrated threshold 2
        self.Threshold_F1   = 0  # Filtered threshold 1
        self.Threshold_F2   = 0  # Filtered threshold 2

        self.RR_LL          = 0  # RR low limit
        self.RR_HL          = 0  # RR high limit
        self.RR_ML          = 0  # RR missed limit
        self.RR_AVG1        = []  # average of the eight most-recent beats
        self.RR_AVG2        = []  # average of the eight most-recent beats with in the range RR_LL, RR_HL

        self.ecg_signal     = ecg_signal  # ecg signal
        self.mwi_signal     = mwi_signal  # integrated signal
        self.bp_signal      = bp_signal  # bandpassed signal
        self.fs             = fs

        self.win_150ms      = round(0.15 * fs)

        self.FLAG_T_wave    = False

    # Fiducial Marking
    def find_possible_peaks(self):
        FM_peaks = []

        # FFT convolution
        slopes = signal.fftconvolve(self.mwi_signal, np.full((25,), 1) / 25, mode='same')

        # Finding approximate peak locations
        for i in range(round(0.5 * self.fs) + 1, len(slopes) - 1):
            if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                FM_peaks.append(i)

        return FM_peaks

    def update_thresholds(self):
        self.Threshold_I1 = self.NLI + 0.25 * (self.SLI - self.NLI)
        self.Threshold_F1 = self.NLF + 0.25 * (self.SLF - self.NLF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1

    def find_r_peaks(self):
        peaks_FM = self.find_possible_peaks()

        for idx in range(len(peaks_FM)):
            
            win_300ms   = np.arange(max(0, peaks_FM[idx] - self.win_150ms),
                                    min(peaks_FM[idx] + self.win_150ms, len(self.bp_signal) - 1), 1)
            max_ind     = np.argmax(self.bp_signal[win_300ms])
            max_val     = self.bp_signal[win_300ms[max_ind]]
            max_index   = win_300ms[max_ind]

            if max_val > -sys.maxsize:
                self.possible_peaks.append(max_index)

            if idx == 0 or idx > len(self.possible_peaks):  # if new peak found in integration signal

                if self.mwi_signal[peaks_FM[idx]] >= self.Threshold_I1:  # if a peak is a signal peak

                    self.SLI        = 0.125 * self.mwi_signal[peaks_FM[idx]] + 0.875 * self.SLI

                    if self.possible_peaks[idx] > self.Threshold_F1:  # if new peak found in bandpass filtered signal

                        self.SLF    = 0.125 * self.bp_signal[idx] + 0.875 * self.SLF
                        self.signal_peaks.append(self.possible_peaks[idx])

                    else:
                        self.NLF    = 0.125 * self.bp_signal[idx] + 0.875 * self.NLF
                elif ((self.mwi_signal[peaks_FM[idx]] > self.Threshold_I2 and
                       self.mwi_signal[peaks_FM[idx]] < self.Threshold_I1) or
                      (self.mwi_signal[peaks_FM[idx]] < self.Threshold_I2)):
                    
                    self.NLI        = 0.125 * self.mwi_signal[peaks_FM[idx]] + 0.875 * self.NLI
                    self.NLF        = 0.125 * self.bp_signal[idx] + 0.875 * self.NLF

            else:
                self.RR_1           = np.diff(peaks_FM[max(0, idx - 8):idx + 1]) / self.fs
                rr_one_mean         = np.mean(self.RR_1)
                self.RR_AVG1.append(rr_one_mean)
                limit_factor        = rr_one_mean

                if idx >= 8:
                    # calculate RR limits and rr_avg_two
                    for RR in self.RR_1:
                        if self.RR_LL < RR < self.RR_HL:
                            self.RR_AVG2.append(RR)
                            if len(self.RR_AVG2) == 9:
                                self.RR_AVG2.pop(0)
                                limit_factor = np.mean(self.RR_AVG2)

                # RR limits
                if len(self.RR_2) == 8 or idx < 8:
                    self.RR_LL      = 0.92 * limit_factor
                    self.RR_HL      = 1.16 * limit_factor
                    self.RR_ML      = 1.66 * limit_factor

                # For irregular heart rates
                if self.RR_1[-1] < self.RR_LL or self.RR_1[-1] > self.RR_ML:
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                # Searchback
                current_RR      = self.RR_1[-1]
                sb_window       = round(current_RR * self.fs)
                sb_max_idx      = -1
                sb_max_value    = -sys.maxsize

                if current_RR > self.RR_ML:
                    lower_bound = peaks_FM[idx] - sb_window + 1
                    upper_bound = peaks_FM[idx] + 1

                    for i in range(lower_bound, upper_bound):
                        if self.mwi_signal[i] > self.Threshold_I1 and self.mwi_signal[i] > sb_max_value:
                            sb_max_value    = self.mwi_signal[i]
                            sb_max_idx      = i

                    if sb_max_idx != -1:
                        # update the thresholds
                        self.SLI = 0.25 * self.mwi_signal[sb_max_idx] + 0.75 * self.SLI
                        self.update_thresholds()

                        upper_bound     = sb_max_idx - self.win_150ms
                        lower_bound     = min(len(self.bp_signal) - 1, sb_max_idx)
                        sb_max_idx2     = -1
                        sb_max_value    = -sys.maxsize

                        for i in range(lower_bound, upper_bound):
                            if self.bp_signal[i] > self.Threshold_F1 and self.bp_signal[i] > sb_max_value:
                                sb_max_value    = self.bp_signal[i]
                                sb_max_idx2     = i

                        if sb_max_idx2 != -1:
                            # QRS complex detected
                            if self.bp_signal[sb_max_idx2] > self.Threshold_F2:
                                self.SLF = 0.25 * self.bp_signal[sb_max_idx2] + 0.75 * self.SLF
                                self.update_thresholds()
                                self.signal_peaks.append(sb_max_idx2)

                    # T wave detection
                    if self.mwi_signal[peaks_FM[idx]] >= self.Threshold_I1:
                        if 0.20 < current_RR < 0.36 and idx > 0:
                            current_slope = max(np.diff(self.mwi_signal[
                                                       peaks_FM[idx] - round(self.fs * self.win_150ms / 2):
                                                       peaks_FM[idx] + 1]))
                            previous_slope = max(np.diff(self.mwi_signal[
                                                        peaks_FM[idx - 1] - round(self.fs * self.win_150ms / 2):
                                                        peaks_FM[idx - 1] + 1]))
                            if current_slope < 0.5 * previous_slope:
                                self.FLAG_T_wave = True
                                self.NLI = 0.125 * self.mwi_signal[peaks_FM[idx]] + 0.875 * self.NLI

                        if not self.FLAG_T_wave:
                            self.SLI = 0.125 * self.mwi_signal[peaks_FM[idx]] + 0.875 * self.SLI
                            if self.possible_peaks[idx] > self.Threshold_F1:
                                self.SLF = 0.125 * self.bp_signal[idx] + 0.875 * self.SLF
                                self.signal_peaks.append(self.possible_peaks[idx])
                            else:
                                self.NLF = 0.125 * self.bp_signal[idx] + 0.875 * self.NLF
                        elif ((self.mwi_signal[peaks_FM[idx]] > self.Threshold_I1 and
                               self.mwi_signal[peaks_FM[idx]] < self.Threshold_I2) or
                              (self.mwi_signal[peaks_FM[idx]] < self.Threshold_I1)):
                            
                            self.NLI = 0.125 * self.mwi_signal[peaks_FM[idx]] + 0.875 * self.NLI
                            self.NLF = 0.125 * self.bp_signal[idx] + 0.875 * self.NLF

            self.update_thresholds()
            self.FLAG_T_wave = False

        # searching in ECG signal to increase accuracy
        for i in np.unique(self.signal_peaks):
            i = int(i)
            window = round(0.2 * self.fs)
            left_limit = i - window
            right_limit = min(i + window + 1, len(self.ecg_signal))
            max_value = -sys.maxsize
            max_index = -1
            for j in range(left_limit, right_limit):
                if j > 0:
                    if self.ecg_signal[j] > max_value:
                        max_value = self.ecg_signal[j]
                        max_index = j

            self.r_peaks.append(max_index)

        # Correct for T wave peaks
        corrected_peaks = []
        max_val         = np.max(self.ecg_signal[self.r_peaks])
        for p in self.r_peaks:
            if self.ecg_signal[p] > max_val/2:
                corrected_peaks.append(p)
        
        self.r_peaks    = corrected_peaks
        return self.r_peaks
    