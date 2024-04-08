import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import scipy.fftpack as fftpack

class Corpus: 
    
    def __init__(self, LIB, OHCO, book_dir):
        self.LIB = LIB
        self.OHCO = OHCO
        self.book_dir = book_dir
        
        
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
        values_fft = fftpack.fft(raw_values, padding_len)
        low_pass_size = low_pass_size * (1 + padding_factor)
        keepers = values_fft[:low_pass_size]

        # Preserve frequency domain structure
        modified_spectrum = list(keepers) \
            + list(np.zeros((x_reverse_len * (1+padding_factor)) - (2*low_pass_size) + 1)) \
            + list(reversed(np.conj(keepers[1:(len(keepers))])))

        # Strip padding
        inverse_values = fftpack.ifft(modified_spectrum)
        inverse_values = inverse_values[:x_reverse_len]

        transformed_values = np.real(tuple(inverse_values))
        return transformed_values        
        
    def DCT(raw_values, 
                  low_pass_size=5, 
                  x_reverse_len=100,
                  dct_type=3):
        if low_pass_size > len(raw_values):
            raise ValueError("low_pass_size must be less than or equal to the length of raw_values input vector")
        values_dct = fftpack.dct(raw_values, type = dct_type) # 2 or 3 works well
        keepers = values_dct[:low_pass_size]
        padded_keepers = list(keepers) + list(np.zeros(x_reverse_len - low_pass_size))
        dct_out = fftpack.idct(padded_keepers)
        return dct_out        

class SyuzhetBook:
    
    # Set this at the class level before creating instances
    corpus:Corpus = None
    
    def __init__(self, book_id):
        self.book_id = book_id
        self.book_title = self.corpus.LIB.loc[book_id].title
        
    def compute_sentiment(self, norm='standard'):
        """Get a book and apply compute various sentiments by lexicon and aggregation"""

        bag = self.corpus.OHCO[:3] # Sentences -- NOTE THIS CAN EASILY FAIL; NEED TO PUT SENTENCE LEVEL IN CONFIG

        # Grab the book tokens
        book_file = self.corpus.LIB.loc[self.book_id].book_file
        TOKENS = pd.read_csv(f"{self.corpus.book_dir}/{book_file}")\
            .set_index(self.corpus.OHCO).sort_index()

        # Dict of various computed sentiments per bag
        S = {} 

        # Group by sentences
        G = TOKENS.fillna(0).groupby(bag)

        ### HANDLE NRC

        # Count positives and negatives for each sentence
        P = G['nrc_positive'].sum()
        N = G['nrc_negative'].sum()    

        # Apply Logit Scale
        S['nrc_logit'] = np.log(P + .5) - np.log(N + .5) 

        # Apply Relative Proportional Difference (mean)
        S['nrc_rpd'] = ((P - N) / (P + N)).fillna(0)

        # Apply simple sum and mean
        S['nrc_sum'] = G['nrc_polarity'].sum()
        S['nrc_mean'] = G['nrc_polarity'].mean() 

        ### HANDLE OTHERS

        # Apply sum and mean to others
        for salex in 'bing syu gi'.split():
            S[f'{salex}_sum'] = G[f'{salex}_sentiment'].sum()
            S[f'{salex}_mean'] = G[f'{salex}_sentiment'].mean()

        # Combine into single dataframe
        SENTS = pd.concat(S.values(), axis=1).sort_index()
        SENTS.columns = S.keys()

        # Normalize the data
        if norm == 'standard':
            SENTS = (SENTS - SENTS.mean()) / SENTS.std()    
        elif norm == 'minmax':
            SENTS = 2 * (SENTS - SENTS.min()) / (SENTS.max() - SENTS.min()) - 1
        else: 
            pass

        self.SENTS = SENTS    
        
    def visualize_raw(self):

        plot_cfg = dict(
            figsize=(25, 3 * len(self.SENTS.columns) * 1), 
            legend=False, 
            fontsize=16)

        fig, axes = plt.subplots(len(self.SENTS.columns), 1)

        for i, col in enumerate(self.SENTS.columns):
            plot_title = self.book_title + ': ' + col + ' (raw)'
            self.SENTS[col].plot(ax=axes[i], **plot_cfg);
            axes[i].set_title(plot_title, fontsize=24)
            axes[i].set_xlabel('')

        plt.tight_layout()        
        
    def visualize_smooth(self, 
                 dct=True, 
                 low_pass_size=5, 
                 x_reverse_len=100):

        S_methods = self.SENTS.columns.tolist()

        fig, axes = plt.subplots(len(S_methods), 1)

        plot_cfg = dict(
            figsize=(25, 5 * len(S_methods) * 1), 
            legend=False, 
            fontsize=16)

        for i, S_method in enumerate(S_methods):

            X = self.SENTS[S_method].values

            if dct:
                method="DCT"
                X = Transforms.DCT(X, low_pass_size=low_pass_size, x_reverse_len=x_reverse_len)
            else:
                method="FFT"
                X = Transforms.FFT(X, low_pass_size=low_pass_size, x_reverse_len=x_reverse_len, padding_factor = 1)

            # Scale Range
            X = (X - X.mean()) / X.std()

            plot_title = f"{self.book_title}: {S_method} ({method})"

            pd.Series(X).plot(**plot_cfg, ax=axes[i]);
            axes[i].set_title(plot_title, fontsize=20) # title of plot

        plt.tight_layout()
        
    def plot_rolling(self, 
                         salex='bing',
                         win_type='cosine', 
                         win_div=3,
                         agg='mean',
                         norm=None):

        window = round(self.SENTS.shape[0]/win_div)
        plot_title = self.book_title + f'  (rolling; div={win_div}, w={window}, {salex}, agg={agg})'
        ax = self.SENTS[f"{salex}_{agg}"].rolling(window, win_type=win_type).mean().plot(figsize=(25,5))
        ax.set_title(plot_title, fontsize=20)

    def visualize_single(self, 
                         salex='bing',
                         agg='mean',
                         low_pass_size=5, 
                         x_reverse_len=100, 
                         dct_type=3,
                         norm=None
                        ):

        # book_title = LIB.loc[book_id].title
        # sents = compute_sentiment(book_id, norm=norm)

        s_method = f"{salex}_{agg}"
        X = self.SENTS[s_method].values
        X = Transforms.DCT(X, low_pass_size=low_pass_size, x_reverse_len=x_reverse_len, dct_type=dct_type)
        X = (X - X.mean()) / X.std()

        plot_title = f"{self.book_title}: {s_method} (DCT)"
        ax = pd.Series(X).plot(figsize=(25, 5), legend=False, fontsize=16);
        ax.set_title(plot_title, fontsize=20)
        plt.tight_layout()

    def plotitall(self, norm=None, dct=True, low_pass_sizes=[3,5]):
        self.compute_sentiment()
        self.visualize_raw()
        for lps in low_pass_sizes:
            self.visualize_smooth(dct=dct, low_pass_size=lps)
        # self.visualize_smooth(dct=True, low_pass_size=5)        