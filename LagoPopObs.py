import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import sys
from pathlib import Path

# Useful variables
win_len_list = ("256", "512", "1024", "2048", "4056", "8192")
overlap_list = [str(k) for k in range(5,100,5)]
default_lago_vars = [True, "980","2800","1024", "75", "2048", "90", "54"]

class LagoPopObsUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        # Window title
        self.title("lagoPopObs")
        # Geometry of window
        self.geometry("1000x700")
        self.resizable(False, False)
        # self.rowconfigure(10, weight=1)
        # Variables
        self.dir_input = tk.StringVar(self,"")
        self.dir_output = tk.StringVar(self, "")
        self.wavelet_filt = tk.BooleanVar(self, default_lago_vars[0])
        self.fmin = tk.StringVar(self,default_lago_vars[1])
        self.fmax = tk.StringVar(self,default_lago_vars[2])
        self.win_fft = tk.StringVar(self,default_lago_vars[3])
        self.ovlp_fft = tk.StringVar(self,default_lago_vars[4])
        self.win_env = tk.StringVar(self,default_lago_vars[5])
        self.ovlp_env = tk.StringVar(self,default_lago_vars[6])
        self.n_features = tk.StringVar(self,default_lago_vars[7])
        # Test
        # filename = filedialog.askdirectory(initialdir=Path(sys.executable).parent)
        # Entries
        # Apply wavelet filtering or not?
        check_wlt_filt = tk.Checkbutton(self, text="Apply wavelet fintering?", variable=self.wavelet_filt)
        check_wlt_filt.pack(fill=tk.X)
        # Minimum frequency
        lab_fmin = ttk.Label(text="Lowest frequency:")
        lab_fmin.pack(fill=tk.X)
        entry_fmin = tk.Entry(self, textvariable=self.fmin)
        entry_fmin.pack(fill=tk.X)
        # Maximum frequency
        lab_fmax = ttk.Label(text="Highest frequency:")
        lab_fmax.pack(fill=tk.X)
        entry_fmax = tk.Entry(self, textvariable=self.fmax)
        entry_fmax.pack(fill=tk.X)
        # Window length for Short-Term Fourier Transform on sound
        lab_win_fft = ttk.Label(text="Window length for STFT of sound:")
        lab_win_fft.pack(fill=tk.X)
        combo_win_fft = ttk.Combobox(self, textvariable=self.win_fft)
        combo_win_fft["values"] = win_len_list
        combo_win_fft["state"] = "readonly"
        combo_win_fft.pack(fill=tk.X)
        # Overlap in percent for Short-Term Fourier Transform on sound
        lab_ovlp_fft = ttk.Label(text="Overlap (in %) for STFT of sound:")
        lab_ovlp_fft.pack(fill=tk.X)
        combo_ovlp_fft = ttk.Combobox(self, textvariable=self.ovlp_fft)
        combo_ovlp_fft["values"] = overlap_list
        combo_ovlp_fft["state"] = "readonly"
        combo_ovlp_fft.pack(fill=tk.X)
        # Window length for Short-Term Fourier Transform on sound
        lab_win_env = ttk.Label(text="Window length for STFT of envelope:")
        lab_win_env.pack(fill=tk.X)
        combo_win_env = ttk.Combobox(self, textvariable=self.win_env)
        combo_win_env["values"] = win_len_list
        combo_win_env["state"] = "readonly"
        combo_win_env.pack(fill=tk.X)
        # Overlap in percent for Short-Term Fourier Transform on envelope
        lab_ovlp_env = ttk.Label(text="Overlap (in %) for STFT of envelope:")
        lab_ovlp_env.pack(fill=tk.X)
        combo_ovlp_env = ttk.Combobox(self, textvariable=self.ovlp_env)
        combo_ovlp_env["values"] = overlap_list
        combo_ovlp_env["state"] = "readonly"
        combo_ovlp_env.pack(fill=tk.X)
        # Number of n_features
        lab_n_features = ttk.Label(text="Number of features to extract:")
        lab_n_features.pack(fill=tk.X)
        entry_n_features = tk.Entry(self, textvariable=self.n_features)
        entry_n_features.pack(fill=tk.X)




if __name__ == "__main__":
    win = LagoPopObsUI()
    win.mainloop()
