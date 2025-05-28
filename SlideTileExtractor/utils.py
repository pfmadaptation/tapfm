import openslide
import numpy as np
import cv2
import pdb

msk_aperio_20x_mpp = 0.50185

#Find mult and level to get the mpp to equal that
def normalize_msk20x(slide):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = msk_aperio_20x_mpp/mpp
    level = 0
    return level, mult

# Get marker from slides
# Code from Vijay
# https://github.com/MSKCC-Computational-Pathology/slidereader/blob/master/modules/gen_coords/extract_molecular_annotations.py
def detect_marker(thumb, mult):
    ksize = int(max(1, mult))
    #ksize = 1
    img = cv2.GaussianBlur(thumb, (5,5), 0)
    hsv_origimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Extract marker
    black_marker = cv2.inRange(hsv_origimg, np.array([0, 0, 0]), np.array([180, 255, 125])) # black marker
    blue_marker = cv2.inRange(hsv_origimg, np.array([90, 30, 30]), np.array([130, 255, 255])) # blue marker
    green_marker = cv2.inRange(hsv_origimg, np.array([40, 30, 30]), np.array([90, 255, 255])) # green marker
    mask_hsv = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    mask_hsv = cv2.erode(mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
    mask_hsv = cv2.dilate(mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize*3,ksize*3)))
    if np.count_nonzero(mask_hsv) > 0:
        return mask_hsv
    else:
        return None
