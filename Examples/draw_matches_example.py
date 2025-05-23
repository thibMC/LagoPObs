# Modified from: https://stackoverflow.com/questions/20358217/opencv-drawing-matches-in-top-and-bottom-configuration

# Librairies
try:
    from tools import utils, filtering, spectro, image_matching
except:
    from LagoPObs.tools import utils, filtering, spectro, image_matching

import cv2

# parameters
input_dir = "LagoPObs/Examples"
wlt_filt = "Yes"  # Wavelet filtering?
f_filt = [950, 2800]  # frequency bandwidth
wlen = 281  # Window length
ovlp = 75  # overlap spectro
wlen_env = 706
ovlp_env = 90  # overlap spectro enveloppe
n_matches = 53  # number of matches
detector_methode = "ORB custom"  # Feature extraction algorithm

# Perform analysis
# filter the WAV filenames in the input directory
list_wavs = utils.filter_wavs(input_dir)
list_wavs = list_wavs[1:]  # only the two ptarmigan sounds here

# Import the sounds, transform them in float64 arrays
# and normalize them by their RMS
list_arr, sf = utils.import_wavs(list_wavs, input_dir, f_filt[1])
# Bandpass filtering
list_arr_filt = [filtering.butterfilter(a, sf, f_filt) for a in list_arr]
# Wavelet filtering if wlt_fit == "Yes"
if wlt_filt == "Yes":
    list_arr_filt = [filtering.wlt_denoise(a) for a in list_arr_filt]
arr_filt = utils.pad_signals(list_arr_filt)
# Drawing the spectrograms
spectros = [
    spectro.draw_specs(
        a,
        int(wlen),
        int(ovlp),
        int(wlen_env),
        int(ovlp_env),
        sf,
        f_filt,
    )
    for a in arr_filt
]
# Transformation in 8 bits images for compatibility with OpenCV
spec_8bits = [image_matching.transfo_8bits(s) for s in spectros]
# Initiate the detector and matcher
detector, matcher = image_matching.feature_detector_matcher(name=detector_methode)

# We rotate only to display the example as it is more convenient to have the images on top of each, otherwise no rotation in the normal pipeline
rotate = True
if rotate:
    im1 = cv2.rotate(spec_8bits[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
    im2 = cv2.rotate(spec_8bits[1], cv2.ROTATE_90_COUNTERCLOCKWISE)

# Detect keypoints and descrptors
kp1, ds1 = detector.detectAndCompute(im1, None)
kp2, ds2 = detector.detectAndCompute(im2, None)

# Matching
matches = matcher.match(ds1, ds2)

# Draw matches:
im_matches = cv2.drawMatches(
    im1,
    kp1,
    im2,
    kp2,
    matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
# undo pre-rotation
if rotate:
    im_matches = cv2.rotate(im_matches, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(im_matches)
plt.show()
