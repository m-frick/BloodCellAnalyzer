#import
import cv2
import numpy as np
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from skimage import measure


def rbc(bs, border = True, tf_factor = 0.4):
    """Returns markers and bs for image segmentation"""
    #bs stands for blood smear
    #bs = cv2.imread(path)

    dim =(2000, 1500)
    bs_resized = cv2.resize(bs, dim , interpolation = cv2.INTER_AREA)
    #converts image to gray
    gray = cv2.cvtColor(bs_resized, cv2.COLOR_BGR2GRAY)

    #image smoothing helps reducing the noise (here Gaussian filter)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #setting threshold (here binary inverse + OTSU threshold)
    ret, thresh = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #morphological gradient is the difference between dilation and erosion of an image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    #Finding the contours and fill them
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(opening, [cnt], 0, 255, -1)

    #clearing the borders
    if border == True:
        opening = clear_border(opening)

    #background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, tf_factor * dist_transform.max(),
                                 255, 0)
    sure_fg = np.uint8(sure_fg)
    #unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)

    #set markers(labels):
    ##############################################################
    ret, markers = cv2.connectedComponents(sure_fg)

    #background pixel value 0 --> problem
    #watershed would consider this as unknown
    #add 10(or any other int) to all labels so that background is not 0
    markers = markers + 10

    #mark the region of unknown with 0
    markers[unknown == 255] = 0

    #apply the watershed with the markers to the original image
    markers = cv2.watershed(bs_resized, markers)
    bs_resized[markers == -1] = [255, 0, 0]
    ##############################################################

    return bs_resized, markers

    #def rbc_seg_plt(path,):
    #bs, markers = rbc(path)
    #(x, y, ax=None, **plt_kwargs):
    #if ax is None:
    #ax = plt.gca()
    #ax.plot(x, y, **plt_kwargs) ## example plot here
    #return(ax)


def cell_crop(image):
    bs_resized, markers = rbc(image)

    regions = measure.regionprops(markers, intensity_image=bs_resized)
    #Region Of Interest
    list_ROI = [regions[0].intensity_image.tolist()]
    for prop in regions:
        h, w, c = prop.intensity_image.shape
        if h <= 155 and w <= 155 and h >= 90 and w >= 90:
            list_ROI.append(prop.intensity_image.tolist())

    return list_ROI
