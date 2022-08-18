from ast import main
from fileinput import filename
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

cv2.namedWindow("Blue mask")
cv2.moveWindow("Blue mask", 0, screen_height // 3)
cv2.namedWindow("Blue result")
cv2.moveWindow("Blue result", 0, (screen_height // 3) * 2)

cv2.namedWindow("Yellow mask")
cv2.moveWindow("Yellow mask", screen_width // 2, screen_height // 3)
cv2.namedWindow("Yellow result")
cv2.moveWindow("Yellow result", screen_width // 2, (screen_height // 3) * 2)
cv2.destroyAllWindows()

morphic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
joining_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 20))
polishing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

hsv_color_ranges = {
    'blue': {
        'lower': np.array([90, 50, 70]),
        'upper': np.array([128, 255, 255])
    },
    'yellow': {
        'lower': np.array([25, 50, 70]),
        'upper': np.array([35, 255, 255])
    }
}


def main(img):
    # image loading

    # crop out black border added to every image in the dataset, border is 140 pixels wide
    height = img.shape[0]
    width = img.shape[1]

    img = img[150:height - 150, 150:width - 150]

    height -= 150
    width -= 150

    start_point = (width - 1300, height - 350)
    end_point = (width - 1, height - 1)
    img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), -1)

    # conversion to HSV color space to prepare for color thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # thresholding
    blue_mask = cv2.inRange(hsv, hsv_color_ranges['blue']['lower'],
                            hsv_color_ranges['blue']['upper'])
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_OPEN,
                                 morphic_kernel,
                                 iterations=5)
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_CLOSE,
                                 joining_kernel,
                                 iterations=3)
    blue_mask = cv2.morphologyEx(blue_mask,
                                 cv2.MORPH_CLOSE,
                                 polishing_kernel,
                                 iterations=6)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.08 * cnt_len, True)
        if len(cnt) == 3 and cv2.contourArea(cnt) > 500:
            triangles.append(cnt)

    bounding_boxes = [cv2.boundingRect(triangle) for triangle in triangles]
    blue_result = cv2.bitwise_and(img, img, mask=blue_mask)
    for box in bounding_boxes:
        blue_result = cv2.rectangle(blue_result, box, color=(0, 250, 0), thickness= 10)
    
    #blue_result = cv2.drawContours(blue_result, bounding_boxes, -1, (0, 250, 0), 10)
    

    yellow_mask = cv2.inRange(hsv, hsv_color_ranges['yellow']['lower'],
                              hsv_color_ranges['yellow']['upper'])
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                 cv2.MORPH_OPEN,
                                 morphic_kernel,
                                 iterations=5)
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                 cv2.MORPH_CLOSE,
                                 joining_kernel,
                                 iterations=3)
    yellow_mask = cv2.morphologyEx(yellow_mask,
                                 cv2.MORPH_CLOSE,
                                 polishing_kernel,
                                 iterations=6)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.08 * cnt_len, True)
        if len(cnt) == 3 and cv2.contourArea(cnt) > 500:
            triangles.append(cnt)
    
    bounding_boxes = [cv2.boundingRect(triangle) for triangle in triangles]
    yellow_result = cv2.bitwise_and(img, img, mask=yellow_mask)
    for box in bounding_boxes:
        yellow_result = cv2.rectangle(yellow_result, box, color=(0, 250, 0), thickness= 10)
    
    #yellow_result = cv2.drawContours(yellow_result, triangles, -1, (0, 250, 0), 10)
    # resize to a quarter of a screen
    correction_factor = 1.

    if width > height:
        correction_factor = (screen_width / 2.) / width
    else:
        correction_factor = (screen_height / 2.) / height

    img = cv2.resize(
        img, (int(width * correction_factor), int(height * correction_factor)),
        0, 0, cv2.INTER_AREA)

    blue_mask = cv2.resize(
        blue_mask,
        (int(width * correction_factor), int(height * correction_factor)), 0,
        0, cv2.INTER_AREA)

    blue_result = cv2.resize(
        blue_result,
        (int(width * correction_factor), int(height * correction_factor)), 0,
        0, cv2.INTER_AREA)

    yellow_mask = cv2.resize(
        yellow_mask,
        (int(width * correction_factor), int(height * correction_factor)), 0,
        0, cv2.INTER_AREA)

    yellow_result = cv2.resize(
        yellow_result,
        (int(width * correction_factor), int(height * correction_factor)), 0,
        0, cv2.INTER_AREA)

    # show results
    cv2.imshow("Original", img)
    cv2.imshow("Blue mask", blue_mask)
    cv2.imshow("Yellow mask", yellow_mask)
    cv2.imshow("Blue result", blue_result)
    cv2.imshow("Yellow result", yellow_result)
    cv2.waitKey()


if __name__ == "__main__":
    Tk().withdraw()
    paths = askopenfilenames(title="Select file",
                             initialdir="./images/")
    imgs = []
    for path in paths:
        imgs.append(cv2.imread(path))
    cv2.waitKey()

    for i in range(len(paths)):
        print(paths[i])
        main(imgs[i])