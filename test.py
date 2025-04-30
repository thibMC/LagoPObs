# https://scikit-image.org/docs/stable/auto_examples/transform/plot_matching.html#sphx-glr-auto-examples-transform-plot-matching-py

import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (
    corner_harris,
    corner_subpix,
    corner_peaks,
    plot_matched_features,
)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

best_spec_8bits = [transfo_8bits(s) for s in spectrogrammes[param_best[0][0]]]

img_orig_gray = best_spec_8bits[0]
img_warped_gray = best_spec_8bits[1]

# extract corners using Harris' corner measure
coords_orig = corner_peaks(
    corner_harris(img_orig_gray), threshold_rel=0.001, min_distance=1
)
coords_warped = corner_peaks(
    corner_harris(img_warped_gray), threshold_rel=0.001, min_distance=5
)

plt.imshow(img_orig_gray)
plt.plot(coords_orig[:, 1], coords_orig[:, 0], "or")
plt.show()

# determine sub-pixel corner position
coords_orig_subpix = corner_subpix(img_orig_gray, coords_orig, window_size=9)
coords_warped_subpix = corner_subpix(img_warped_gray, coords_warped, window_size=9)


def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext : window_ext + 1, -window_ext : window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


def match_corner(coord, window_ext=5):
    r, c = np.round(coord).astype(np.intp)
    window_orig = img_orig_gray[
        r - window_ext : r + window_ext + 1, c - window_ext : c + window_ext + 1
    ]
    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    weights = np.dstack((weights, weights))
    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_warped:
        window_warped = img_warped_gray[
            cr - window_ext : cr + window_ext + 1, cc - window_ext : cc + window_ext + 1
        ]
        SSD = np.sum(weights * (window_orig - window_warped) ** 2)
        SSDs.append(SSD)
    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return coords_warped_subpix[min_idx]


# find correspondences using simple weighted sum of squared differences
src = []
dst = []
for coord in coords_orig_subpix:
    src.append(coord)
    dst.append(match_corner(coord))

src = np.array(src)
dst = np.array(dst)


# estimate affine transform model using all coordinates
model = AffineTransform()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
model_robust, inliers = ransac(
    (src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100
)
outliers = inliers == False


# compare "true" and estimated transform parameters
print("Ground truth:")
print(
    f"Scale: ({tform.scale[1]:.4f}, {tform.scale[0]:.4f}), "
    f"Translation: ({tform.translation[1]:.4f}, "
    f"{tform.translation[0]:.4f}), "
    f"Rotation: {-tform.rotation:.4f}"
)
print("Affine transform:")
print(
    f"Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), "
    f"Translation: ({model.translation[0]:.4f}, "
    f"{model.translation[1]:.4f}), "
    f"Rotation: {model.rotation:.4f}"
)
print("RANSAC:")
print(
    f"Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), "
    f"Translation: ({model_robust.translation[0]:.4f}, "
    f"{model_robust.translation[1]:.4f}), "
    f"Rotation: {model_robust.rotation:.4f}"
)

# visualize correspondence
fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

inlier_idxs = np.nonzero(inliers)[0]
plot_matched_features(
    img_orig_gray,
    img_warped_gray,
    keypoints0=src,
    keypoints1=dst,
    matches=np.column_stack((inlier_idxs, inlier_idxs)),
    ax=ax[0],
    matches_color="b",
)
ax[0].axis("off")
ax[0].set_title("Correct correspondences")

outlier_idxs = np.nonzero(outliers)[0]
plot_matched_features(
    img_orig_gray,
    img_warped_gray,
    keypoints0=src,
    keypoints1=dst,
    matches=np.column_stack((outlier_idxs, outlier_idxs)),
    ax=ax[1],
    matches_color="r",
)
ax[1].axis("off")
ax[1].set_title("Faulty correspondences")

plt.show()
