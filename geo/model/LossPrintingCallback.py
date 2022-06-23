import tensorflow as tf

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, gGuessr, saveFolder, modelNumber):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.model = gGuessr
        self.modelNumber = modelNumber
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        self.model.loss = round(float(logs['loss']), 3)

    def on_train_end(self, logs={}):
        print("Training sucessfull!")
        self.model.save(self.saveFolder, self.modelNumber)
