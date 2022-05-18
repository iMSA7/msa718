#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:02:34 2020

@author: muhannad
"""

import pickle


class ModelHandler:
    def __init__(self, model, file):
        self.model = model
        self.file_path = file

    def load(self):
        return pickle.load(open(self.file_path, 'rb'))

    def save(self):
        pickle.dump(self.model, open(self.file_path, 'wb'))
        print(f"saved to {self.file_path}")