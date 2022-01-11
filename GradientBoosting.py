# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:07:09 2021

@author: Yeeun Kim
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
    
class GradientBoosting():
    
    def __init__(self, n_models, learning_rate=0.1):
        
        self.init_model = DecisionTreeClassifier
        self.proc_model = DecisionTreeRegressor
        self.n_models = n_models
        self.models = []
        self.learning_rate = learning_rate
        
        
    def fit(self, X, y):
        
        self.groups = y.unique()
        self.groups.sort()
        y = (y == self.groups[1]).astype(int)
                
        init_model = self.init_model(max_depth=3)
        init_model.fit(X, y)
        self.models.append(init_model)
        
        ypred = init_model.predict_proba(X)[:,1]
        r = y - ypred

        for b in range(self.n_models - 1):
            proc_model = self.proc_model(max_depth=3)
            proc_model.fit(X, r)
            self.models.append(proc_model)
            
            ypred = ypred + self.learning_rate * proc_model.predict(X)
            r = y - ypred
            
    def predict(self, X, cutoff=0.5):
        
        ypred = self.models[0].predict_proba(X)[:,1]
        
        for m in self.models[1:]:
            ypred = ypred + self.learning_rate * m.predict(X)
        
        pred = np.full_like(ypred, fill_value = self.groups[0])
        pred[ypred > cutoff] = self.groups[1]
        
        return pred