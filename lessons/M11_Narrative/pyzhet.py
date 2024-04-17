import pandas as pd
import numpy as np
import scipy as sp
import scipy.fft as fft
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

class Transforms:
    
    def FFT(raw_values, 
               low_pass_size=3, 
               x_reverse_len=100,  
               padding_factor=2, 
               scale_values=False, 
               scale_range=False):

        if low_pass_size > len(raw_values):
            sys.exit("low_pass_size must be less than or equal to the length of raw_values input vector")

        raw_values_len = len(raw_values)
        padding_len = raw_values_len * padding_factor

        # Add padding, then fft
        values_fft = fft.fft(raw_values, padding_len)
        low_pass_size = low_pass_size * (1 + padding_factor)
        keepers = values_fft[:low_pass_size]

        # Preserve frequency domain structure
        modified_spectrum = list(keepers) \
            + list(np.zeros((x_reverse_len * (1+padding_factor)) - (2*low_pass_size) + 1)) \
            + list(reversed(np.conj(keepers[1:(len(keepers))])))

        # Strip padding
        inverse_values = fft.ifft(modified_spectrum)
        inverse_values = inverse_values[:x_reverse_len]

        transformed_values = np.real(tuple(inverse_values))
        return transformed_values        
        
    def DCT(raw_values, 
                  low_pass_size=5, 
                  x_reverse_len=100,
                  dct_type=3):
        
        if low_pass_size > len(raw_values):
            raise ValueError("low_pass_size must be less than or equal to the length of raw_values input vector")
        
        values_dct = fft.dct(raw_values, type = dct_type) # 2 or 3 works well
        
        keepers = values_dct[:low_pass_size]
        
        padded_keepers = list(keepers) + list(np.zeros(x_reverse_len - low_pass_size))
        
        dct_out = fft.idct(padded_keepers)
        
        return dct_out


class SyuzhetBook:
    
    def __init__(self, sent_vec:pd.Series, book_title:str):
        
        self.sent_vec = sent_vec
        self.book_title = book_title
        
    def plot_raw(self):

        plot_cfg = dict(
            figsize=(25, 5), 
            legend=False, 
            fontsize=16)
        
        self.sent_vec.plot(**plot_cfg)
        
    def plot_smooth(self, 
                 method='DCT', 
                 low_pass_size=6, 
                 x_reverse_len=100):

        plot_cfg = dict(
            figsize = (25, 5), 
            legend = False, 
            fontsize = 16,
            title = f"{self.book_title} {method}"
        )

        if method == "DCT":
            X = Transforms.DCT(self.sent_vec.values, low_pass_size=low_pass_size, x_reverse_len=x_reverse_len)
        
        elif method == "FFT":
            X = Transforms.FFT(self.sent_vec.values, low_pass_size=low_pass_size, x_reverse_len=x_reverse_len, padding_factor = 1)
            
        else:
            raise(ValueError("Argument 'method' must be DCT or FFT."))

        # Scale Range
        X = (X - X.mean()) / X.std()

        pd.Series(X).plot(**plot_cfg);
        
    def plot_rolling(self, 
                         win_type='cosine', 
                         win_div=3,
                         norm=None):
        
        window = round(self.sent_vec.shape[0]/win_div)
        plot_title = self.book_title + f'  (rolling; div={win_div}, w={window})'
        self.sent_vec.rolling(window, win_type=win_type).mean().plot(figsize=(25,5), title=plot_title)