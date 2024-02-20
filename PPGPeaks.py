class PPG_Peaks:

    def __init__(self, PPG_signal, R_peak):
        self.R_peak     = R_peak
        self.signal     = PPG_signal

    def findMin(self):

        first_index = self.signal.index[0]
        PPG     = self.signal[self.R_peak-first_index:]
        idxmin  = PPG.idxmin()
        return idxmin
    
    def findMaxMin(self):

        idxmin  = self.findMin()
        first_index = self.signal.index[0]
        PPG     = self.signal[idxmin-first_index:idxmin-first_index+400]
        idxmax  = PPG.idxmax()
        
        return idxmax, idxmin
