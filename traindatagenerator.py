# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:00:03 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import numpy as np
import tensorflow as tf
import librosa

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, x_train, y_train, batch_size=4, shuffle=True, frame_length=20):
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.frame_length = frame_length
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.y_train) / self.batch_size))
    
    def __getitem__(self, index): 
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]   
         
        x_train = []
        for path in np.array(self.x_train)[indexes]: 
            data = self.data_load(path) 
            x_train.append(data)
         
        x_train = np.array(x_train)  
        y_train = np.array(self.y_train)[indexes]   
         
        return x_train, y_train
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
     
    def data_load(self, path):
        
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
        mfccs = tf.reshape(mfccs, shape=[20, 3])

        return mfccs 