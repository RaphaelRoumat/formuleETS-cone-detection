from ast import main
from fileinput import filename
from os import remove
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames

import cv2
import numpy as np
from screeninfo import get_monitors

#from os import environ # remove comment if crash on get_monitors()

screen = get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

cv2.namedWindow("Original")
cv2.moveWindow("Original", screen_width // 4, 0)

cv2.namedWindow("1")
cv2.moveWindow("1", 0, screen_height // 3)
cv2.namedWindow("2")
cv2.moveWindow("2", 0, (screen_height // 3) * 2)

cv2.namedWindow("3")
cv2.moveWindow("3", screen_width // 2, screen_height // 3)
cv2.namedWindow("4")
cv2.moveWindow("4", screen_width // 2, (screen_height // 3) * 2)
cv2.destroyAllWindows()

morphic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
joining_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 20))
polishing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

hsv_color_ranges = {
    'blue': {
        'lower': np.array([90, 100, 70]),
        'upper': np.array([128, 255, 255])
    },
    'yellow': {
        'lower': np.array([25, 50, 70]),
        'upper': np.array([35, 255, 255])
    }
}


def findRegionsOfInterest(img_medianBlurred):
    # conversion to HSV color space to prepare for color thresholding
    hsv = cv2.cvtColor(img_medianBlurred, cv2.COLOR_BGR2HSV)

    # thresholding
    blue_mask = cv2.inRange(hsv, hsv_color_ranges['blue']['lower'],
                            hsv_color_ranges['blue']['upper'])
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_OPEN,
                                 morphic_kernel,
                                 iterations=5)
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_CLOSE,
                                 morphic_kernel,
                                 iterations=5)
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_DILATE,
                                 morphic_kernel,
                                 iterations=2)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bounding_box = [cv2.boundingRect(contour) for contour in contours]

    blue_enlarged_box = []
    
    for box in bounding_box:
        x, y, w, h = box[0], box[1], box[2], box[3]
        new_w = int(w * 1.4)
        new_h = int(h * 1.9)
        x_displacemet, y_displacement = (new_w - w, new_h - h)
        new_x, new_y = x - (x_displacemet // 2), y - (y_displacement // 2)
        
        new_box = (new_x, new_y, new_w, new_h)
        blue_enlarged_box.append(new_box)

    yellow_mask = cv2.inRange(hsv, hsv_color_ranges['yellow']['lower'],
                              hsv_color_ranges['yellow']['upper'])
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                   cv2.MORPH_OPEN,
                                   morphic_kernel,
                                   iterations=5)
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                   cv2.MORPH_CLOSE,
                                   morphic_kernel,
                                   iterations=5)
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                   cv2.MORPH_DILATE,
                                   morphic_kernel,
                                   iterations=2)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bounding_box = [cv2.boundingRect(contour) for contour in contours]

    yellow_enlarged_box = []
    
    for box in bounding_box:
        x, y, w, h = box[0], box[1], box[2], box[3]
        new_w = int(w * 1.4)
        new_h = int(h * 1.9)
        x_displacemet, y_displacement = (new_w - w, new_h - h)
        new_x, new_y = x - (x_displacemet // 2), y - (y_displacement // 2)
        
        new_box = (new_x, new_y, new_w, new_h)
        yellow_enlarged_box.append(new_box)

    return blue_enlarged_box, yellow_enlarged_box


def removeOverlappingRegions(regions):
        
    i = 0
    while  i < len(regions):
        region = regions[i]
        x1, y1, x2, y2 = region[0], region[1], region[0] + region[2], region[1] + region[3]
        
        for j in range(0, len(regions)):
            if i != j:
                region_p = regions[j]
                x1_p, y1_p, x2_p, y2_p = region_p[0], region_p[1], region_p[0] + region_p[2], region_p[1] + region_p[3]
                if x1 < x1_p and y1 < y1_p and x2 > x2_p and y2 > y2_p:
                    regions.pop(i)
                    i -=1
                    break
        i += 1

    return regions

def main(img):
    pass
    
if __name__ == "__main__":
    Tk().withdraw()
    paths = askopenfilenames(title="Select file", initialdir="./images/")
    imgs = []
    for path in paths:
        imgs.append(cv2.imread(path))
    cv2.waitKey()

    for i in range(len(paths)):
        print(paths[i])
        blue_regions, yellow_regions = findRegionsOfInterest(cv2.medianBlur(imgs[i], 3))
        blue_regions = removeOverlappingRegions(blue_regions)
        yellow_regions = removeOverlappingRegions(yellow_regions)
        
        for region in blue_regions:
            imgs[i] = cv2.rectangle(imgs[i], region, (0, 255, 0), thickness= 5)
        
        cv2.imshow("1", imgs[i])
        cv2.waitKey()
        # TODO ajouter la edge detection seulement sur les régions d'intérêt en utilisant le draw contours pour enlever les lignes inutiles
        main(imgs[i])