import tensorflow as tf

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    '''
    Custom callback used to save model at the end of a training batch.
    '''

    def __init__(self, gGuessr, saveFolder, modelNumber):
        '''
        gGuess: Instance of Geoguessr
        saveFolder: Name of floder to save to
        modelNumber: A number to add to saved model file name.
        Used to find how far into training a model was saved
        '''
        super(LossAndErrorPrintingCallback, self).__init__()
        self.model = gGuessr
        self.modelNumber = modelNumber
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        '''
        Update model loss every few epochs
        '''
        self.model.loss = round(float(logs['loss']), 3)

    def on_train_end(self, logs={}):
        """
        Save model at the end of training
        """
        print("Training sucessfull!!")
        self.model.save(self.saveFolder, self.modelNumber)
