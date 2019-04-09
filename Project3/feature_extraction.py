import image_splitter
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import numpy as np
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb


def HOG(image):

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

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

def converttogreyscale(image_array):
    image_array=cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    return image_array


def write_kp(frame):
    kp,des=getSIFT(frame)
    kp_frame = cv2.drawKeypoints(frame, kp[:],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp_frame

def getSIFT(frame):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(frame,None)
    return kp,des


pixel_array=image_splitter.loadimage("./agriculture.tif")

list_of_regions=image_splitter.chop_img(pixel_array,10)

radius = 3
n_points = 8 * radius
METHOD = 'uniform'


mine = image_splitter.loadimage('./mine_train.JPG')
mine=converttogreyscale(mine)

trees=image_splitter.loadimage('./trees.png')
trees=converttogreyscale(trees)
agr1=image_splitter.loadimage('./agriculture_train1.JPG')
agr1=converttogreyscale(agr1)
agr2=image_splitter.loadimage('./agriculture_train2.JPG')
agr2=converttogreyscale(agr2)

agr3=image_splitter.loadimage('./agriculture_train3.JPG')
agr3=converttogreyscale(agr3)



refs = {
    "agriculture1":local_binary_pattern(agr1, n_points, radius, METHOD),
    "agriculture2": local_binary_pattern(agr2, n_points, radius, METHOD),
    "agriculture3": local_binary_pattern(agr3, n_points, radius, METHOD),
    'trees': local_binary_pattern(trees, n_points, radius, METHOD),
    #'mine': local_binary_pattern(mine, n_points, radius, METHOD),
    }

# classify rotated textures
for frame in list_of_regions:
    frame=converttogreyscale(frame)
    match_result = match(refs, frame)
    print(' match result: ',match_result)
    if match_result is 'mine':
        image_splitter.displayimage(frame,str(frame))

