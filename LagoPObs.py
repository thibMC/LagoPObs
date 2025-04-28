import sys
import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showwarning, showerror, askyesno
from tkinter import filedialog
from tkinter import font
# import sv_ttk
# import darkdetect

# variables
# List of choices for window length
win_len_list = ("256", "512", "1024", "2048", "4056", "8192")
# List of choices for overlap
overlap_list = [str(k) for k in range(5, 100, 5)]
# Default values for rock ptarmigan
default_lago_vars = ["Yes", "980", "2800", "1024", "75", "2048", "90", "54"]
# Option for wavelet filtering
wlt_filt_list = ["Yes", "No"]


class LagoPopObsUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        # Prepare the grid
        # Rows
        for i in range(18):
            self.grid_rowconfigure(i, weight=0)
        # Columns
        for j in range(2):
            self.grid_columnconfigure(j, weight=1, uniform="same_group")
            # self.grid_columnconfigure(j, weight=1)
        # Default configurations for each element of the grid
        default_grid = {"padx": 2, "pady": 2, "sticky": "nsew"}
        default_separator = {"padx": 10, "pady": 10, "sticky": "nsew"}
        # Window title
        self.title("lagoPopObs")
        # Geometry of window
        self.geometry("1000x850")
        self.resizable(True, False)
        # Variables
        self.dir_input = tk.StringVar(self, os.getcwd())
        self.dir_output = tk.StringVar(self, os.getcwd())
        self.wavelet_filt = tk.StringVar(self, default_lago_vars[0])
        self.fmin = tk.StringVar(self, default_lago_vars[1])
        self.fmax = tk.StringVar(self, default_lago_vars[2])
        self.win_fft = tk.StringVar(self, default_lago_vars[3])
        self.ovlp_fft = tk.StringVar(self, default_lago_vars[4])
        self.win_env = tk.StringVar(self, default_lago_vars[5])
        self.ovlp_env = tk.StringVar(self, default_lago_vars[6])
        self.n_features = tk.StringVar(self, default_lago_vars[7])
        # Welcome text
        lab_welcome = ttk.Label(
            self,
            text="Welcome to the Lagopède Population Observator, in short LagoPObs, developped by Reef pulse SAS for the LPO.\nThis software was designed initially to cluster rock ptarmigan males \n according to each type of their vocalisations,\nbased on the difference of spectrograms.\nPotentially, it could be used to separate other type of sounds\nby tinkering the analysis parameters but with no guarantee of results. \n",
        )
        lab_welcome.grid(row=0, column=0, columnspan=2, **default_separator)
        # Separator
        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )
        lab_folders = ttk.Label(self, text="Folders")
        lab_folders.grid(row=2, column=0, columnspan=2, **default_separator)
        #  ttk.Separator(self, orient="horizontal").grid(row=1, column=1, columnspan=3, sticky='nse')
        # Input folder
        lab_dir_input = ttk.Label(text="Input folder containing wav files to study:")
        lab_dir_input.grid(row=3, column=0, columnspan=2, **default_grid)
        button_dir_input = ttk.Button(
            self, text="Select input folder", command=self.input_folder
        )
        button_dir_input.grid(row=4, column=0, **default_grid)
        self.text_dir_input = ttk.Text(self, wrap="word", height=1)
        self.text_dir_input.insert("1.0", self.dir_input.get())
        self.text_dir_input.config(state="disabled")
        self.text_dir_input.grid(row=4, column=1, **default_grid)
        # Output folder
        lab_dir_output = ttk.Label(text="Output folder where the results will be saved")
        lab_dir_output.grid(row=5, column=0, columnspan=2, **default_grid)
        button_dir_output = ttk.Button(
            self, text="Select ouput folder", command=self.output_folder
        )
        button_dir_output.grid(row=6, column=0, **default_grid)
        self.text_dir_output = ttk.Text(self, wrap="word", height=1)
        self.text_dir_output.insert("1.0", self.dir_output.get())
        self.text_dir_output.config(state="disabled")
        self.text_dir_output.grid(row=6, column=1, **default_grid)
        # Separator
        ttk.Separator(self, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=20
        )
        lab_params = ttk.Label(text="Analysis parameters")
        lab_params.grid(row=8, column=0, columnspan=2, **default_separator)
        # Apply wavelet filtering or not?
        lab_wlt_filt = ttk.Label(text="Wavelet filtering:")
        lab_wlt_filt.grid(row=9, column=0, **default_grid)
        combo_wlt_filt = ttk.Combobox(self, textvariable=self.wavelet_filt)
        combo_wlt_filt["values"] = wlt_filt_list
        combo_wlt_filt["state"] = "readonly"
        combo_wlt_filt.grid(row=9, column=1, **default_grid)
        # Minimum frequency
        lab_fmin = ttk.Label(text="Lowest frequency:")
        lab_fmin.grid(row=10, column=0, **default_grid)
        entry_fmin = ttk.Entry(self, textvariable=self.fmin)
        entry_fmin.grid(row=10, column=1, **default_grid)
        # Maximum frequency
        lab_fmax = ttk.Label(text="Highest frequency:")
        lab_fmax.grid(row=11, column=0, **default_grid)
        entry_fmax = ttk.Entry(self, textvariable=self.fmax)
        entry_fmax.grid(row=11, column=1, **default_grid)
        # Window length for Short-Term Fourier Transform on sound
        lab_win_fft = ttk.Label(text="Window length for STFT of sound:")
        lab_win_fft.grid(row=12, column=0, **default_grid)
        combo_win_fft = ttk.Combobox(self, textvariable=self.win_fft)
        combo_win_fft["values"] = win_len_list
        combo_win_fft["state"] = "readonly"
        combo_win_fft.grid(row=12, column=1, **default_grid)
        # Overlap in percent for Short-Term Fourier Transform on sound
        lab_ovlp_fft = ttk.Label(text="Overlap (in %) for STFT of sound:")
        lab_ovlp_fft.grid(row=13, column=0, **default_grid)
        combo_ovlp_fft = ttk.Combobox(self, textvariable=self.ovlp_fft)
        combo_ovlp_fft["values"] = overlap_list
        combo_ovlp_fft["state"] = "readonly"
        combo_ovlp_fft.grid(row=13, column=1, **default_grid)
        # Window length for Short-Term Fourier Transform on sound
        lab_win_env = ttk.Label(text="Window length for STFT of envelope:")
        lab_win_env.grid(row=14, column=0, **default_grid)
        combo_win_env = ttk.Combobox(self, textvariable=self.win_env)
        combo_win_env["values"] = win_len_list
        combo_win_env["state"] = "readonly"
        combo_win_env.grid(row=14, column=1, **default_grid)
        # Overlap in percent for Short-Term Fourier Transform on envelope
        lab_ovlp_env = ttk.Label(text="Overlap (in %) for STFT of envelope:")
        lab_ovlp_env.grid(row=15, column=0, **default_grid)
        combo_ovlp_env = ttk.Combobox(self, textvariable=self.ovlp_env)
        combo_ovlp_env["values"] = overlap_list
        combo_ovlp_env["state"] = "readonly"
        combo_ovlp_env.grid(row=15, column=1, **default_grid)
        # Number of n_features
        lab_n_features = ttk.Label(text="Number of features to extract:")
        lab_n_features.grid(row=16, column=0, **default_grid)
        entry_n_features = ttk.Entry(self, textvariable=self.n_features)
        entry_n_features.grid(row=16, column=1, **default_grid)
        # Button to validate the parameters and proceed to analysis
        button_proceed = ttk.Button(
            self, text="Validate and proceed to analysis", command=self.validate_proceed
        )
        button_proceed.grid(row=17, column=0, columnspan=2, **default_separator)

    def input_folder(self):
        folder = filedialog.askdirectory(initialdir=self.dir_input.get())
        if folder != ():
            files = os.listdir(folder)
            wav_check = any([f.endswith(".wav") for f in files])
            if wav_check:
                self.text_dir_input.configure(state="normal")
                self.text_dir_input.delete("0.0", tk.END)
                self.text_dir_input.insert("1.0", folder)
                self.dir_input.set(folder)
                self.text_dir_output.configure(state="disabled")
            else:
                showwarning(
                    title="Wrong input folder!",
                    message="No WAV files in your input folder!",
                )

    def output_folder(self):
        folder = filedialog.askdirectory(initialdir=self.dir_output.get())
        if folder != ():
            self.text_dir_output.configure(state="normal")
            self.text_dir_output.delete("0.0", tk.END)
            self.text_dir_output.insert("1.0", folder)
            self.dir_output.set(folder)
            self.text_dir_input.configure(state="disabled")

    def validate_proceed(self):
        # Config font size
        font1 = font.Font(name="TkCaptionFont", exists=True)
        font1.config(size=11)
        # List with all problems
        list_problems = []
        # List with all warnings
        list_warnings = []
        # Check that input folder exists and contains WAV
        if os.path.isdir(self.dir_input.get()):
            files_input = os.listdir(self.dir_input.get())
            wav_check = any([f.endswith(".wav") for f in files_input])
            if not wav_check:
                list_problems.append("- Your input folder does not contain WAV files.")
        else:
            list_problems.append("- Your input folder does not exist.")
        # Check that output folder exists
        if not os.path.isdir(self.dir_output.get()):
            list_problems.append("- Your output folder does not exist.")
        # Check that the minimum frequency, maximum frequency and number of features are integers
        vars_value = [v.get() for v in [self.fmin, self.fmax, self.n_features]]
        vars_name = ["lowest frequency", "highest frequency", "number of features"]
        for v in zip(vars_value, vars_name):
            try:
                float_n = float(v[0])
            except ValueError:
                list_problems.append("- The %s is not a number." % (v[1]))
            else:
                if int(float_n) != float_n:
                    list_warnings.append(
                        "- The %s is a decimal, it will be rounded to the closest integer."
                        % (v[1])
                    )
        if list_problems:
            list_problems += list_warnings
            list_problems = [
                "We cannot continue as several problems occured:"
            ] + list_problems
            showerror(title="Wrong inputs!", message="\n".join(list_problems))
        else:
            if list_warnings:
                list_warnings = [
                    "Several decimals were detected in the parameters:"
                ] + list_warnings
                showwarning(title="Floats in inputs!", message="\n".join(list_warnings))
            param_list = [
                "Input folder: ",
                "Output folder: ",
                "Apply wavelet filtering: ",
                "Lowest frequency: ",
                "Highest frequency: ",
                "Window length: ",
                "Overlap (%): ",
                "Envelope window length: ",
                "Envelope overlap (%): ",
                "Number of features: ",
            ]
            param_values = [
                self.dir_input.get(),
                self.dir_output.get(),
                self.wavelet_filt.get(),
                str(round(float(self.fmin.get()))),
                str(round(float(self.fmax.get()))),
                self.win_fft.get(),
                self.ovlp_fft.get(),
                self.win_env.get(),
                self.ovlp_env.get(),
                str(round(float(self.n_features.get()))),
            ]
            param_valid = [p[0] + p[1] for p in zip(param_list, param_values)]
            param_valid = [
                "Do you wish to proceed with the following parameters?"
            ] + param_valid
            answer = askyesno(
                title="Validation of parameters", message="\n".join(param_valid)
            )
            print(answer)

# # This is where the magic happens
# sv_ttk.set_theme(darkdetect.theme())

if __name__ == "__main__":
    win = LagoPopObsUI()
    win.mainloop()
