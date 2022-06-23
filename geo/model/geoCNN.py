from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D

from geo.data.utils import dataGen
from geo.model.LossPrintingCallback import LossAndErrorPrintingCallback


class GeoGuessrCNN:
    def __init__(self,
                 model=None,
                 loss=-1,
                 inputShape=(3, 300, 600, 3),
                 outputShape=124,
                 modelOptimizer=tf.keras.optimizers.Adam()):
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
        y_pred = []
        ln = len(imgFiles)
        for idx, (xx, yy) in enumerate(dataGen(imgFiles, dataDir, batchSize=1, infinite=False)):
            yp = self.model.predict(xx)[0]
            y_pred.append(yp)
            if idx % checkPoint == 0:
                print("Evaluated {} out of {} points".format(idx, ln))

        return y_pred

    def save(self, saveFolder, modelNumber=0):
        if self.loss == -1:
            print("Cannot save untrained model!")
        else:
            print("\nSaving model {} with loss {} at {}".format(modelNumber,
                                                                self.loss,
                                                                saveFolder))
            self.model.save(saveFolder + '/model_{}_{}.h5'.format(self.loss,
                                                                  modelNumber))

    @classmethod
    def load(cls, loadFile):
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        loss = float(modelFile.split('_')[1])
        print("Loaded model loss {}".format(loss))
        return cls(model=model, loss=loss)
