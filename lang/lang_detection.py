import os

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from pytesseract import pytesseract

import utils
from decode import decode
from draw import drawPolygons, drawBoxes
from nms import nms


def text_detection(image, east, min_confidence, width, height):
    # load the input image and grab the image dimensions
    image = cv2.imread(image)
    orig = image.copy()
    (origHeight, origWidth) = image.shape[:2]

    remH = origHeight % height
    remW = origWidth % width
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)

    # ratioWidth = origWidth / float(newW)
    # ratioHeight = origHeight / float(newH)
    ratioWidth = 1
    ratioHeight = 1

    # resize the image and grab the new image dimensions
    # image = cv2.resize(image, (newW, newH))
    image = image[0:origHeight-remH, 0:(origWidth - remW)]
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))


    # NMS on the the unrotated rects
    confidenceThreshold = min_confidence
    nmsThreshold = 0.4

    # decode the blob info
    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    ##########################################################

    # functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]
    functions = [nms.felzenszwalb.nms]
    print("[INFO] Running nms.boxes . . .")

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                 nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)

        if len(indicies):
            drawrects = np.array(rects)[indicies]

            for (x, y, w, h) in drawrects:
                box = orig[y:y+h, x:x+w]
                box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
                # box = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # box = cv2.medianBlur(box, 5)
                kernel = np.ones((5, 5), np.uint8)
                # box = cv2.dilate(box, kernel, iterations=1)
                # box = cv2.erode(box, kernel, iterations=1)
                # box = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel)
                box = cv2.Canny(box, 100, 200)
                cv2.imwrite("box.png", box)
                # text = pytesseract.image_to_string(box, config='-l rus')
                res = pytesseract.image_to_osd("box.png")
                cv2.imshow("box", box)
                cv2.waitKey(0)

            name = function.__module__.split('.')[-1].title()
            print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawrects)))

            drawOn = orig.copy()
            drawBoxes(drawOn, drawrects, ratioWidth, ratioHeight, (0, 255, 0), 2)

        # title = "nms.boxes {}".format(name)
        # cv2.imshow(title,drawOn)
        # cv2.moveWindow(title, 150+i*300, 150)

    # cv2.waitKey(0)



    # convert rects to polys
    # polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)
    #
    # print("[INFO] Running nms.polygons . . .")
    #
    # for i, function in enumerate(functions):
    #
    #     start = time.time()
    #     indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
    #                              nsm_threshold=nmsThreshold)
    #     end = time.time()
    #
    #     indicies = np.array(indicies).reshape(-1)
    #
    #     drawpolys = np.array(polygons)[indicies]
    #     for pol in drawpolys:
    #         cropped = orig[y:y + h, x:x + w]
    #
    #     name = function.__module__.split('.')[-1].title()
    #
    #     print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawpolys)))
    #
    #
    #     drawOn = orig.copy()
    #     drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)
    #
    #     title = "nms.polygons {}".format(name)
    #     # cv2.imshow(title,drawOn)
    #     # cv2.moveWindow(title, 150+i*300, 150)
    #
    # # cv2.waitKey(0)

dirs = os.listdir('/')
# for d in dirs:
d = 'Russia'
files = os.listdir(f'./lang/{d}')
for f in files:
    print(f'Image: {f}')
    text_detection(f'./lang/{d}/{f}','frozen_east_text_detection.pb', 0.5, 320, 320)