# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:20:01 2021

@author: Yeeun kim
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 


class Bagging():
    
    def __init__(self, model, n_models, random_feature = False, n_random_features = None):
        self.model = model
        self.n_models = n_models
        self.models = []
        self.oob_index = []
        self.variable_importance = []
        
        self.random_feature = random_feature
        if random_feature:
            self.features = []
            self.n_random_features = n_random_features
    
    def fit(self, X, y, variable_importance = True):
        self.X = X
        self.y = y
        
        self.n, self.p = X.shape
        
        if not self.random_feature:
            self.features = [list(self.X.columns)] * self.n_models
            
        if self.random_feature and self.n_random_features is None:
            self.n_random_features = round(self.p/2)
        
        for b in range(self.n_models):
            # define base learner
            _model = self.model()
            
            # get bootstrap sample
            _X = X.sample(replace = True, n = self.n)
            _y = y.loc[_X.index]
            
            # random feature
            if self.random_feature:
              _X = _X.sample(replace = False, n = self.n_random_features, axis = 1)
              self.features.append(_X.columns)
            
            # out of bag sample
            self.oob_index.append(set(X.index) - set(_X.index))
            
            # fit base learner
            _model = _model.fit(_X, _y)
            
            # save the base learner
            self.models.append(_model)
            
        if variable_importance:
            self._compute_variable_importance()
            
    def predict(self, X):
        predicted_values = []
        
        for _model, _features in zip(self.models, self.features):
            predicted_values.append(_model.predict(X[_features]))
        
        predicted_values = pd.DataFrame(predicted_values)
        predicted_values = predicted_values.mode().transpose()[0]
        predicted_values.name = 'predicted'
        
        return predicted_values
    
    def _compute_variable_importance(self):
        vi = []
        
        for _model, _oob_index, _features in zip(self.models, self.oob_index, self.features):
            _oob_X = self.X.loc[_oob_index][_features]
            _oob_y = np.array(self.y.loc[_oob_index]).transpose()
            
            _e_b = np.mean(_model.predict(_oob_X) != _oob_y)
                        
            _d_b = []
            for col in self.X.columns:
                if col in _oob_X.columns:
                    _oob_X_p = _oob_X.copy()
                    _oob_X_p[col] = _oob_X_p[col].sample(frac = 1).values
                    _p = np.mean(_model.predict(_oob_X_p) != _oob_y)
                    _d_b.append(_p - _e_b)
                    
                else:
                    _d_b.append(None)
            
            vi.append(_d_b)
            
        vi = pd.DataFrame(vi)
        vi = vi.mean() / vi.std()
        self.variable_importance = vi.sort_values(ascending = False)
