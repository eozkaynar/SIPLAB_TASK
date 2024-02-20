from scipy.ndimage import uniform_filter1d

class Pan_Tompkins:
    #  y(n) = 2y(n-1) - y(n-2) + x(n) - 2x(n-6) + x(n-12)
    def lp_filter(self,x):
        y = x.copy()
        for n in range(len(x)):
            if (n >= 1):
                y[n] += 2*y[n-1]  
            if (n >= 2):
                y[n] -= y[n-2]    
            if (n >= 6):
                y[n] -= 2*x[n-6]  
            if (n >= 12):
                y[n] += x[n-12]   
        return y
    
    def hp_filter(self,x):
    # y(n) = 32x(n-16) - y(n-1) - x(n) + x(n-32)
        y = x.copy()
        for n in range(len(x)):
            if (n >= 1):
                y[n] = -x[n] - y[n-1]
            if (n >= 16):
                y[n] += 32*x[n-16]
            if (n >= 32):
                y[n] += x[n-32]
        max_y = max(max(y),-min(y))
        y = y/max_y
        return y

    def derivative_operator(self,x,fs):
        # y(n) = (1/8) * (2x(n) + x(n-1) - x(n-3) - 2x(n-4))
        y = [0] * len(x)
        for n in range(len(x)):
            if (n == 0):
                y[n] = 2*x[n] / (8*fs)
            elif (n == 1):
                y[n] = (2*x[n] + x[n-1]) / (8*fs)
            elif (n == 2):
                y[n] = (2*x[n] + x[n-1]) / (8*fs)
            elif (n == 3):
                y[n] = (2*x[n] + x[n-1] - x[n-3]) / (8*fs)
            else:
                y[n] = (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4 ]) / (8*fs)

        return y
    def Square(self, x):
        y = [0]*len(x)
        for n in range(len(x)):
            y[n] = x[n]**2
        return y
    def moving_window_integration(self,signal):

        # # Initialize result and window size for integration
        result = signal.copy()
        win_size = round(0.150 * 4000)
        result = uniform_filter1d(signal, win_size)

        return result