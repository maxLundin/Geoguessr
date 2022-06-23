from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D

from geo.data.utils import dataGen
from geo.model.LossPrintingCallback import LossAndErrorPrintingCallback


class GeoGuessrCNN:
    '''
    The class has all the functions required to built, train and test the geoguessr CNN model.
    '''

    def __init__(self,
                 model=None,
                 loss=-1,
                 inputShape=(3, 300, 600, 3),
                 outputShape=124,
                 modelOptimizer=tf.keras.optimizers.Adam()):
        """
        The function is used to load or initialize a new model
        inputShape: Shape of input image set
                    (<numer-of-images>, <image-width>, <image-height>, <RGB-values>)
        outputShape: Number of output classes to predict on
        """
        if model is None:
            convnet = tf.keras.Sequential()
            convnet.add(Conv2D(128, (3, 3), input_shape=inputShape[1:],
                               padding='same', activation='relu'))

            convnet.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            convnet.add(BatchNormalization(momentum=.6))
            convnet.add(MaxPool2D())

            convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            convnet.add(BatchNormalization(momentum=.6))
            convnet.add(MaxPool2D())

            convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            convnet.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            convnet.add(BatchNormalization(momentum=.6))
            convnet.add(MaxPool2D())

            convnet.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
            convnet.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
            convnet.add(BatchNormalization(momentum=.6))
            convnet.add(GlobalMaxPool2D())

            self.model = tf.keras.Sequential()

            self.model.add(Dense(1024, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(64, activation='relu'))

            self.model.add(Dense(outputShape, activation='softmax'))

            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=modelOptimizer, metrics=['categorical_accuracy'])
        else:
            self.model = model

        self.model.summary()
        self.loss = loss

    def fit(self, trainFiles, dataDir, saveFolder, batchSize=10,
            epochs=20,
            plot=False):
        '''
        Used to train the model in datches with each batch going through a fixed number of epochs
        this is done to let the model have sufficient training time on each batch of data.
        trainFiles: list of image file names, eg: <gridNo>+<lat,long>, eg: 60+48.4271513,-110.5611851
        dataDir: Directory that stores combined image files eg: "/dataCombinedSamples/"
        saveFolder: Folder to save trained model to eg: "models"
        '''
        print("Getting data from directory: {}".format(dataDir))
        accuracy = []
        loss = []
        cnt = 0
        for X, y in dataGen(trainFiles, dataDir, batchSize=batchSize, infinite=False):
            callBack = [LossAndErrorPrintingCallback(self, saveFolder, cnt)]
            print("Read {} points. Training now".format(len(X)))
            evalutaion = self.model.fit(X, y,
                                        epochs=epochs, steps_per_epoch=len(X),
                                        callbacks=callBack)
            accuracy += evalutaion.history['categorical_accuracy']
            loss += evalutaion.history['loss']
            cnt += 1
        if plot:
            plt.plot(accuracy)
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.show()

            plt.plot(loss)
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

    def evaluate(self, imgFiles, dataDir, checkPoint=50):
        '''
        Calculates average of distances between target and predicted grids for a list of files
        imgFiles: List of test location image triplet folders. Each element of the list has to look like:
        eg: <gridNo>+<lat,long>
        eg: 60+48.4271513,-110.5611851
        dataDir: Directory that stores combined image files eg: "/dataCombinedSamples/"
        polyGrid: List of polygons that contain make up the USA split into grids.
                  It can be loaded from eg: "infoExtraction/usaPolyGrid.pkl"
        checkPoint: report progress
        '''
        y_pred = []
        ln = len(imgFiles)
        for idx, (xx, yy) in enumerate(dataGen(imgFiles, dataDir, batchSize=1, infinite=False)):
            yp = self.model.predict(xx)[0]
            y_pred.append(yp)
            if idx % checkPoint == 0:
                print("Evaluated {} out of {} points".format(idx, ln))

        return y_pred

    def save(self, saveFolder, modelNumber=0):
        '''
        Saves model to specified folder with specified number along with loss
        saveFolder: Folder to save trained model to eg: "models"
        '''
        if self.loss == -1:
            print("Cannot save untrained model!!!")
        else:
            print("\nSaving model {} with loss {} at {}".format(modelNumber,
                                                                self.loss,
                                                                saveFolder))
            self.model.save(saveFolder + '/model_{}_{}.h5'.format(self.loss,
                                                                  modelNumber))

    @classmethod
    def load(cls, loadFile):
        '''
        Loads model from specified folder with loss
        loadFile: file to load model from eg: "models/restnet_5.738_19.h5"
        '''
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        loss = float(modelFile.split('_')[1])
        print("Loaded model loss {}".format(loss))
        return cls(model=model, loss=loss)
