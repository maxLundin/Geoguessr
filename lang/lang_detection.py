import os
from collections import defaultdict

from langdetect import detect_langs
import numpy as np
import cv2
from pytesseract import pytesseract

import utils
from decode import decode
from nms import nms

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
FUNCTION = nms.felzenszwalb.nms


def box_readability_improval(box, grayscale=True, denoise=False, threshold=False, dilation=False, erosion=False,
                             opening=False, canny=False, sharpen=False):
    try:
        if grayscale:
            box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        if denoise:
            box = cv2.medianBlur(box, 5)
        kernel = np.ones((5, 5), np.uint8)
        if dilation:
            box = cv2.dilate(box, kernel, iterations=1)
        if erosion:
            box = cv2.erode(box, kernel, iterations=1)
        if threshold:
            box = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if opening:
            box = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel)
        if canny:
            box = cv2.Canny(box, 100, 200)
        if sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            box = cv2.filter2D(box, -1, kernel)
        return box
    except:
        return box


def crop_image(image, width, height):
    (origHeight, origWidth) = image.shape[:2]

    remH = origHeight % height
    remW = origWidth % width

    cropH = origHeight - remH
    cropW = origWidth - remW
    image = image[0:cropH, 0:cropW]
    return image, cropH, cropW


def draw_boxes(draw_on, boxes, color=(0, 255, 0), width=1):
    for (x, y, w, h) in boxes:
        startX = int(x)
        startY = int(y)
        endX = int((x + w))
        endY = int((y + h))

        cv2.rectangle(draw_on, (startX, startY), (endX, endY), color, width)


def draw_polys(drawOn, polygons, color=(0, 0, 255), width=1):
    for polygon in polygons:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(drawOn, [pts], True, color, width)

def draw_box(box, title):
    try:
        # box = box_readability_improval(box, sharpen=True)
        cv2.imshow(title, box)
        cv2.waitKey(0)
    except:
        pass


def detect_lang(box):
    box = box_readability_improval(box, threshold=True)
    custom_config = r'-l rus+spa+por+kor+jpn+fra+ell+deu+ara+chi_sim --psm 6'
    try:
        txt = pytesseract.image_to_string(box, config=custom_config)
        if len(txt) > 2:
            print(txt)
            curr_probs = detect_langs(txt)
            return curr_probs
        return []
    except Exception as e:
        print("exception")
        return []


def text_detection(image, detector, width, height, count_boxes=False, count_polys=True, draw_part=False,
                   draw_full=False):
    image = cv2.imread(image)
    orig = image.copy()

    image, cropH, cropW = crop_image(image, width, height)
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet(detector)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences, baggage) = decode(scores, geometry, CONFIDENCE_THRESHOLD)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    res_probs = defaultdict(int)

    indices = nms.boxes(rects, confidences, nms_function=FUNCTION, confidence_threshold=CONFIDENCE_THRESHOLD,
                        nsm_threshold=NMS_THRESHOLD)

    indices = np.array(indices).reshape(-1)

    if count_boxes and len(indices):
        drawrects = np.array(rects)[indices]

        for (x, y, w, h) in drawrects:
            box = orig[y:y + h, x:x + w]
            curr_probs = detect_lang(box)
            for prob in curr_probs:
                res_probs[prob.lang] += prob.prob

            if draw_part:
                draw_box(box, "box")

        print(f"found {len(drawrects)} boxes")

        if draw_full:
            draw_on = orig.copy()
            draw_boxes(draw_on, drawrects, (0, 255, 0), 2)
            title = "full image boxes"
            cv2.imshow(title, draw_on)
            cv2.waitKey(0)

    if count_polys:
        polygons = utils.rects2polys(rects, thetas, offsets)

        print("Polygons counting . . .")

        indices = nms.polygons(polygons, confidences, nms_function=FUNCTION, confidence_threshold=CONFIDENCE_THRESHOLD,
                               nsm_threshold=NMS_THRESHOLD)
        if len(indices):
            indices = np.array(indices).reshape(-1)

            drawpolys = np.array(polygons)[indices]
            print(f"found {len(drawpolys)} boxes")
            for polygon in drawpolys:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                x, y, w, h = cv2.boundingRect(pts)
                mask = np.zeros(orig.shape[:2], np.uint8)
                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

                box = cv2.bitwise_and(orig, orig, mask=mask)
                box = box[y:y + h, x:x + w]
                curr_probs = detect_lang(box)
                for prob in curr_probs:
                    res_probs[prob.lang] += prob.prob

                if draw_part:
                    draw_box(box, "poly")

            if draw_full:
                draw_on = orig.copy()
                draw_polys(draw_on, drawpolys, (0, 255, 0), 2)

                title = "full image polys"
                cv2.imshow(title, draw_on)
                cv2.waitKey(0)
    return res_probs


LANGS = {
    'ara': 'ar',
    'deu': 'de',
    'ell': 'el',
    'fra': 'fr',
    'jap': 'ja',
    'kor': 'ko',
    'por': 'pt',
    'rus': 'ru',
    'spa': 'es'
}

dirs = os.listdir('./lang/')
count_all = 0
count_true = 0
for d in dirs:
    d = 'rus'
    files = os.listdir(f'./lang/{d}')
    for f in files:
        print(f'Image: /{d}/{f}')
        probs = text_detection(f'./lang/{d}/{f}', 'frozen_east_text_detection.pb', 320, 320, draw_part=True)
        print(probs)
        count_all += 1
        m = 0
        l = ''
        for (lang, prob) in probs.items():
            if prob > m:
                l = lang
                m = prob
        if LANGS[d] == l:
            count_true += 1
            print("TRUEEEEEEEEEEEEEEEEEEEEEEEEEE")
print(count_true / count_all)
