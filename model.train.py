# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:03:37 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""
 
import tensorflow as tf
from datasets  import Datasets
from traindatagenerator import DataGenerator
from warnings import simplefilter  
simplefilter(action='ignore', category=FutureWarning)

class ModelTrain(object):

    def __init__(self):

        self.datasets = Datasets()  
        self.PATH = "./datasets/train"
        self.MODEL_SAVED_PATH = "./model/model.h5" 
        self.EPOCHS = 15 
        self.BATCH_SIZE = 2

    def define_model(self, CLASSES=2, INPUT_SHAPE=[20, 3]):
        input = tf.keras.layers.Input(shape=INPUT_SHAPE) 

        m = tf.keras.layers.Conv1D(32, 3, activation='relu')(input)
        m = tf.keras.layers.Conv1D(64, 3, activation='relu')(m)
        m = tf.keras.layers.MaxPooling1D()(m)
        m = tf.keras.layers.Dropout(0.25)(m)
        m = tf.keras.layers.Dense(128, activation='relu')(m)  
        m = tf.keras.layers.LSTM(128, return_sequences=True)(m)
        m = tf.keras.layers.LSTM(128, return_sequences=False)(m) 
        m = tf.keras.layers.Dense(256, activation='relu')(m) 
        m = tf.keras.layers.Flatten()(m)    
        m = tf.keras.layers.Dense(128, activation='relu')(m) 
        m = tf.keras.layers.Dense(CLASSES, activation='sigmoid')(m) 
        
        model = tf.keras.Model(input, m) 

        model.summary() 
        # tf.keras.utils.plot_model(model) 

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['mse', 'accuracy'],
        )

        return model  
    
    def train(self):
        # Get all the train data paths
        x_paths, x_label = self.datasets.get_data_paths(self.PATH)

        # Apply one hot encoder to the data labels
        x_label = self.datasets.one_hot_encoder(x_label=x_label)

        # Get the total class and set the data input shape
        INPUT_SHAPE = [20, 3] 
        CLASSES = len(x_label[0]) 

        # Prepare the training data
        x_y_train = DataGenerator(
            x_paths,  
            x_label,
            batch_size=self.BATCH_SIZE) 

        # Prepare the model
        model = self.define_model(CLASSES=CLASSES, INPUT_SHAPE=INPUT_SHAPE)

        # Train the model
        history = model.fit(
            x_y_train, 
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        # Save the model
        model.save(self.MODEL_SAVED_PATH)
  
if __name__ == '__main__':
    app = ModelTrain()
    app.train()
        
