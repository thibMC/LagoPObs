# LagoPObs: Lagopède Population Observator

This code is © T. Marin-Cudraz, REEF PULSE S.A.S, LPO AuRA, 2025, and it is made available under the MIT license enclosed with the software.

Over and above the legal restrictions imposed by this license, if you use this software for an academic publication then please provide proper attribution. This can be to this code directly:

T. Marin-Cudraz, Y. Bayle, S. Elise, B. Drillat. LagoPObs: Lagopède Population Observator, a software to separate male ptarmigans indivuals from their sounds (2025). github.com/XXXXX.


$$PI_k = \frac{N_{sons}(k)}{N_{sons}(total)} \times \frac{N_{jours}(k)}{N_{jours}(total)}  \qquad(1)$$



# Installation

The software is available as executables (generated using *pyinstaller*) for Windows and Linux that requires no dependencies. macOS can still execute the python code (see below) but will need the installation of the dependencies.

If you want to execute the python code, you would need to install these libraries:

- With pip:

```
pip install numpy scipy scikit-image scikit-learn matplotlib soxr pywavelets pandas opencv-python
```
- With conda:
```
conda install numpy scipy scikit-image scikit-learn matplotlib soxr pywavelets pandas opencv-python
```

# Running the software

## Using executables

No need to install the libraries if you are using the executables.
- For Windows users: clone/download this repo, unzip and go to the folder and double-click on *LagoPObs_windows.exe*. The software may take a few seconds to open.

- For Linux users: clone/download, unzip this repo and go to the folder. Double-click on *LagoPObs_linux*. If the application does not start, then make it executable using the terminal:
    ```
    chmod +x LagoPobs_linux
    ```
    Then you should be able to execute it by double-clicking or using a terminal:
    ```
    .\LagoPobs_linux
    ```

## Executing the python code

The code was tested for Python 3.9, 3.11, 3.12 and 3.13. There should be no problems with 3.10. The only strict requirement is to have a version of scikit-learn that is 1.3 or higher (inclusion of HDBSCAN).
Install the required libraries with *pip* or *conda*.
You will need to clone/download this repo, unzip and go to the folder.
If you want to use the GUI (Graphical User Interface), open a terminal and type:
```
python LagPObs.py
```

If you just want to study and use the underlining code of the analysis without the GUI, e.g. to use it in a pipeline or to test for different configuration at the same time, open and use the *demo_script.py* file.


# Interface description

The GUI is simple with only a few buttons and fields:

![Graphical interface of LagoPObs](Readme/GUI_LagoPObs.png)

To change the folders where the sounds are located, press *Select input folder*, to change the directory where the results will be saved, press *Select output folder*. By default, the location of the program is used. The input folder can have other type of files than WAV inside, it will automatically filter out non-WAV files. It accepts multiple sample rate and bitrate as the sounds will be automatically converted to 64 bits arrays and will be resampled.

Below, you have access to the different analysis parameters. By default, all parameters are set so that the software can discriminate between rock ptarmigan males.

The *Wavelet filtering* option allow to activate/desactivate the option for the software to perform a wavelet denoising scheme based on the stationary wavelet transform (i.e. the "algorithme à trous", see PyWavelets [1] and [2]) using a biorthogonal wavelet. The wavelet cannot be modified at the moment without going into the code, but it should work for a wide range of signals. The sound is first resampled (see below) and bandpass filtered between the lowest and highest filtering, then it is decomposed using the stationary wavelet transform in several levels. For each level, Fisher's kurtosis is calculated and the level is set to 0s if the kurtosis is negative. Then, a soft thresholding is applied using the standard deviation of each particular level as the threshold. The signal is then reconstructed using the modified levels.

The lowest and highest frequency set the frequency bandwidth of the bandpass filtering (an order 10 butterworth filter). The *Highest frequency* parameter is really important as it will also conditioned the new sampling rate at which the sounds will be resampled using the formula:

$$new sampling rate = 2 \times (Highest frequency + 100)$$


100 is added to avoid frequency aliasing at some frequencies of interest. The choice of the *Highest frequency* is thus critical. It also implies that the temporal resolution of your spectrograms will be modified so be sure to check that the window lengths are still appropriate when modifying this parameter.

The characteristics of the spectrograms of the filtered signal and its envelope can be configured using the next 4 parameters: *Window length* and *Overlap* of STFT (Short-Time Fourier Transform) of sound for the spectrogram of the filtered signal and *Window length* and *Overlap* of the STFT for the spectrogram of the envelope. A hamming window is used for both spectrograms and cannot be modified using the interface.

The *Feature extraction algorithm* parameter configures the feature extraction and matching. The goal of those algorithms is to detect critical keypoints in the image that best describe the features of the image. Each keypoint comes with a descriptor, that represents the local characteristics of the image around the keypoint. You can choose a list of classical algorithms taken from OpenCV: ORB [3], Custom ORB is ORB but with parameters tuned specifically for spectrograms of Rock ptarmigan, SIFT [4-5], AKAZE [6], KAZE [7]. ORB or Custom ORB should work fine for most applications and are also the fastest. After the extraction of the keypoints and descriptors





# References

[1] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In 2011 International conference on computer vision (pp. 2564-2571). Ieee.

[2] https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

[1] Lee, G., Gommers, R., Waselewski, F., Wohlfahrt, K., & O'Leary, A. (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237.

[2] https://en.wikipedia.org/wiki/Stationary_wavelet_transform

[3] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In 2011 International conference on computer vision (pp. 2564-2571). Ieee.

[3] https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

[4] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60, 91-110.

[5] https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

[6] https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html

[7] https://docs.opencv.org/4.x/d3/d61/classcv_1_1KAZE.html
