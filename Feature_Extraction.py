class PPG_Points:

    def __init__(self, PPG_signal, R_peak,fs):
        self.R_peak     = R_peak
        self.signal     = PPG_signal
        self.fs         = fs

    def findMin(self):

        first_index     = self.signal.index[0]
        PPG             = self.signal[self.R_peak-first_index:self.R_peak-first_index+int(0.3*self.fs)]
        idxmin          = PPG.idxmin()
        return idxmin
    
    def findMaxMin(self):

        idxmin          = self.findMin()
        first_index     = self.signal.index[0]
        PPG             = self.signal[idxmin-first_index:idxmin-first_index+400]
        idxmax          = PPG.idxmax()
        
        return idxmax, idxmin

class BCG_Waves:
        
    def __init__(self, BCG_signal, R_peak, fs):

        self.R_peak     = R_peak
        self.signal     = BCG_signal
        self.fs         = fs

    def findIWave(self):

        first_index     = self.signal.index[0]
        BCG             = self.signal[self.R_peak-first_index+int(0.05*self.fs):self.R_peak-first_index+int(0.3*self.fs)]
        idxI            = BCG.idxmin()
        return idxI
    
    def findJWave(self):

        first_index     = self.signal.index[0]
        BCG             = self.signal[self.R_peak-first_index+int(0.1*self.fs):self.R_peak-first_index+int(0.3*self.fs)] # between 100ms and 300ms
        idxJ            = BCG.idxmax()
        
        return idxJ
    
    def findKWave(self):
        
        idxJ            = self.findJWave()
        first_index     = self.signal.index[0]
        BCG             = self.signal[idxJ-first_index:idxJ-first_index+int(0.1*self.fs)] # between 100ms and 300ms
        idxK            = BCG.idxmin()
        
        return idxK
    

class BP_Points:

    def __init__(self, BP_signal, R_peak,fs):
        self.R_peak     = R_peak
        self.signal     = BP_signal
        self.fs         = fs

    def findMin(self):

        first_index     = self.signal.index[0]
        BP              = self.signal[self.R_peak-first_index:self.R_peak-first_index+int(0.2*self.fs)]
        idxmin          = BP.idxmin()
        return idxmin
    
    def findMax(self):

        idxmin          = self.findMin()
        first_index     = self.signal.index[0]
        BP              = self.signal[idxmin-first_index:self.R_peak-first_index+int(0.2*self.fs)]
        idxmax          = BP.idxmax()
        
        return idxmax