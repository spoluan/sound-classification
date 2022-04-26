# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:47:25 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN   
"""

from test import ModelTest

class App(object):
    def __init__(self):
        self.test = ModelTest()

    def main(self):
        sound_path = './datasets/test/0_01_9.wav' 
        pred = self.test.predict(sound_path) 
        print('Predicted result:', pred)

if __name__ == '__main__':
    app = App()
    app.main()