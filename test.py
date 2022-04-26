# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:48:49 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import librosa 
import numpy as np
import tensorflow as tf
from pickledump import PickleDumpLoad
import os 
from warnings import simplefilter  
simplefilter(action='ignore', category=FutureWarning)

class ModelTest(object):

    def __init__(self):
        # Test path
        self.TEST_PATH = './datasets/test'

        # Load the saved model
        MODEL_SAVED_PATH = "./model/model.h5" 
        self.model = tf.keras.models.load_model(MODEL_SAVED_PATH)

        # Load the label model

        pickledump = PickleDumpLoad()
        self.label_model = pickledump.load_config('label.mdl')
  
    def load_sound(self, path):

        time_series_x, sampling_rate = librosa.load(path, sr=32000, mono=True) 

        # Extract mfcc
        mfccs = librosa.feature.mfcc(time_series_x, sr=sampling_rate, n_mfcc=20) 

        # Extract melspectogram
        mel = librosa.feature.melspectrogram(y=time_series_x, sr=sampling_rate, n_mels=20,
                                    fmax=8000, win_length=1024, hop_length=320) 

        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        mel_s_scaled_features = np.mean(mel.T, axis=0)

        # Multiply the mfcc and melspectogram: for additional features
        multiply =  np.multiply(mel_s_scaled_features, mfccs_scaled_features)

        mfccs = tf.convert_to_tensor([mfccs_scaled_features, multiply, mel_s_scaled_features])
        mfccs = tf.reshape(mfccs, shape=[1, 20, 3])

        return mfccs

    def check_prediction(self, pred):
        # Get the based label
        label_model = self.label_model

        # Get the prediction
        predicted = tf.argmax(pred, axis=1) 

        # Decode the label
        result = None
        try:
            result = label_model[predicted.numpy()[0]]
        except:
            pass 
 
        return result

    def predict(self, path): 

        # Load sound
        x_test = self.load_sound(path) 

        # Make a prediction
        pred = self.model.predict(x_test) 

        # Check the prediction
        result = self.check_prediction(pred)

        return result

    def main(self):
        
        test_paths = os.listdir(self.TEST_PATH)
        
        for x in test_paths: 
            path = os.path.join(self.TEST_PATH, x)
            
            # Make a prediction
            result = self.predict(path)

            # Print result
            print(f'{path} = {result}') 
  
if __name__ == '__main__':
    app = ModelTest()
    app.main()


