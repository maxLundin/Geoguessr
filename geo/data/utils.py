import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img


def dataGen(fileNames, dataDir, batchSize=10, infinite=True):
    totalBatches = len(fileNames) / batchSize
    counter = 0
    while (True):
        prev = batchSize * counter
        nxt = batchSize * (counter + 1)
        counter += 1
        yield readData(fileNames[prev:nxt], dataDir)
        if counter >= totalBatches:
            if infinite:
                counter = 0
            else:
                break


def readData(fileNames, dataDir, classes, numclasses):
    X = np.array(list(map(lambda x: np.array(load_img(dataDir + x)), fileNames)))
    y = tf.keras.utils.to_categorical(list(map(lambda x: classes[x.split("/")[0]], fileNames)),
                                      num_classes=numclasses)
    return X, y
