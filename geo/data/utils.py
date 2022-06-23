import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img


def dataGen(fileNames, dataDir, batchSize=10, infinite=True):
    """
    Takes a list of location folder names and ouputs
    a list of input image vectors and ouput categorical grid vector pairs in batches.
    The function is essentially used as a generator that calls readData in datches.
    Inifinit: Tells the function to stop or keep going once the list of file names has been iterated through
    """
    totalBatches = len(fileNames) / batchSize
    counter = 0
    while (True):
        prev = batchSize * counter
        nxt = batchSize * (counter + 1)
        counter += 1
        yield self.readData(fileNames[prev:nxt], dataDir)
        if counter >= totalBatches:
            if infinite:
                counter = 0
            else:
                break


def readData(fileNames, dataDir, classes, numclasses):
    '''
        Takes a list of location folder names and ouputs a list of input image vectors and ouput categorical grid vector pairs.
        fileNames should look like: 60+48.4271513,-110.5611851
        '''
    X = np.array(list(map(lambda x: np.array(load_img(dataDir + x)), fileNames)))
    y = tf.keras.utils.to_categorical(list(map(lambda x: classes[x.split("/")[0]], fileNames)),
                                      num_classes=numclasses)
    return X, y
