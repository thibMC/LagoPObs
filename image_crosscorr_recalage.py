plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(img2)
plt.show()

from skimage.registration import phase_cross_correlation

# pixel precision first
shift, error, diffphase = phase_cross_correlation(img, img2)

plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(img2[int(shift[0]) :, int(shift[1]) :])
plt.show()


kp_best = [best_detector.detectAndCompute(l, None) for l in [img, img2]]
dist_images = [
    distance_matches(best_matcher, kp_best[i], kp_best[j], n_features_best)
    for i in range(len(kp_best))
    for j in range(len(kp_best))
]

kp_best = [
    best_detector.detectAndCompute(l, None)
    for l in [img, img2[int(shift[0]) :, int(shift[1]) :]]
]
dist_images = [
    distance_matches(best_matcher, kp_best[i], kp_best[j], n_features_best)
    for i in range(len(kp_best))
    for j in range(len(kp_best))
]

shift, error, diffphase = phase_cross_correlation(img2, img)
kp_best = [
    best_detector.detectAndCompute(l, None)
    for l in [img[int(shift[0]) :, int(shift[1]) :], img2]
]
dist_images = [
    distance_matches(best_matcher, kp_best[i], kp_best[j], n_features_best)
    for i in range(len(kp_best))
    for j in range(len(kp_best))
]
