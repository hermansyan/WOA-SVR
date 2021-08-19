# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:19:52 2021

@author: MUZANNI
"""
import numpy as np

class SVR_Model:
    def __init__(self, cLR=0.05, C_value=100, epsilon=0.0005, lamda=0.1, sigma=0.3):
        self.cLR = cLR
        self.C_value = C_value
        self.epsilon = epsilon
        self.lamda = lamda
        self.sigma = sigma
        self.solutions = np.array([cLR, C_value, epsilon, lamda, sigma])
        
        self.errors = 0
        self.last_iteration = 0
        self.alpha_function = []
        self.training_data = []
        self.validation_result = []
        