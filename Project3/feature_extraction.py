import cv2
import image_splitter

def converttogreyscale(image_array):
    image_array=cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    return image_array


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

plt.rcParams['font.size'] = 9

# settings for LBP, for more info see
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
#
radius = 2
n_points = 2 * radius
METHOD = 'uniform'

# lpb is the local binary pattern computed for each pixel in the image
def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# The Kullback-Leibler divergence is a measure of how one probability distribution
# is different from a second, reference probability distribution.
# These probability distributions are the histograms computed from the LBP
# KL(p,q) = 0 means p and q distributions are identical.
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# refs is an array reference LB patterns for various classes (brick, grass, wall)
# img is an input image
# match() gives the best match by comparing the KL divergence between the histogram
# of the img LBP and the histograms of the refs LBPs.
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name
def run_LBP():
    agri1=image_splitter.loadimage("./data_750/other/other1.jpg")
    agri1=converttogreyscale(agri1)
    agri5=image_splitter.loadimage("./data_750/agriculture/agr5.jpg")
    agri5=converttogreyscale(agri5)
    agri9=image_splitter.loadimage("./data_750/agriculture/agr9.jpg")
    agri9=converttogreyscale(agri9)


    refs = {
            'agr1': local_binary_pattern(agri1, n_points, radius, METHOD),
        'agr5': local_binary_pattern(agri5, n_points, radius, METHOD),
        'agr9': local_binary_pattern(agri9, n_points, radius, METHOD),
    }

    # classify rotated textures
    #print('Rotated images matched against references using LBP:')
    #print('original: brick, rotated: 30deg, match result: ',
     #     match(refs, rotate(agri1, angle=30, resize=False)))
    #print('original: brick, rotated: 70deg, match result: ',
    #      match(refs, rotate(agri5, angle=70, resize=False)))
    #print('original: grass, rotated: 145deg, match result: ',
    #      match(refs, rotate(agri9, angle=145, resize=False)))

    # plot histograms of LBP of textures
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                           figsize=(9, 6))
    plt.gray()

    ax1.imshow(agri1)
    ax1.axis('off')
    hist(ax4, refs['agr1'])
    ax4.set_ylabel('Percentage')

    ax2.imshow(agri5)
    ax2.axis('off')
    hist(ax5, refs['agr5'])
    ax5.set_xlabel('Uniform LBP values')

    ax3.imshow(agri9)
    ax3.axis('off')
    hist(ax6, refs['agr9'])

    plt.show()


from skimage.feature import hog
from skimage import  exposure

def run_hog():
    image = image_splitter.loadimage("./data_750/agriculture/agr9.jpg")

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    print(fd.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.figure(2)
    plt.plot(fd)

    plt.show()

from scipy import ndimage as ndi

from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
other1=image_splitter.loadimage("./data_750/other/other1.jpg")
other1=converttogreyscale(other1)
other1 = img_as_float(other1)[shrink]
agri5=image_splitter.loadimage('./data_750/agriculture/agr5.jpg')
agri5=converttogreyscale(agri5)
agri5 = img_as_float(agri5)[shrink]
agri9=image_splitter.loadimage('./data_750/agriculture/agr9.jpg')
agri9=converttogreyscale(agri9)
agri9 = img_as_float(agri9)[shrink]
image_names = ('other1', 'agri5', 'agri9')
images = (other1, agri5, agri9)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(other1, kernels)
ref_feats[1, :, :] = compute_feats(agri5, kernels)
ref_feats[2, :, :] = compute_feats(agri9, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: brick, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(other1, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: brick, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(other1, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: grass, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(agri5, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
