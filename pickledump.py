# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:47:20 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import pickle
import os

class PickleDumpLoad(object): 

    def __init__(self): 
        self.address = f'./model'
        
    def save_config(self, obj, filename):  
        with open(os.path.join(self.address, filename), 'wb') as config_f:
            pickle.dump(obj, config_f, protocol=4)
        print('{} saved.' . format(os.path.join(self.address, filename)))
        
    def load_config(self, filename):  
        with open(os.path.join(self.address, filename), 'rb') as f_in:
            obj = pickle.load(f_in)
        return obj 