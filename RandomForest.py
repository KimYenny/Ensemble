# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:08:03 2022

@author: Yeeun Kim
"""

from sklearn import tree
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

class RandomForestClassifier():
    
    def __init__(self, n_models = 101, n_random_features = None):
        
        self.n_models = n_models
        self.__models = []
        self.__oob_index = []
        self.variable_importance = []
        
        self.features = []
        self.n_random_features = n_random_features
    
    def fit(self, X, y, variable_importance = False):
        self.X = X
        self.y = y
        
        self.n, self.p = X.shape
        
        self.groups = y.unique()
        self.groups.sort()
        self.k = len(self.groups)
        
        # mtry
        if self.n_random_features is None:
            self.n_random_features = int(np.log2(self.p) + 1)
        
        # Training
        for b in range(self.n_models):
            # define base learner
            _model = tree.DecisionTreeClassifier()
            
            # get bootstrap sample
            _X = X.sample(replace = True, n = self.n)
            _y = y.loc[_X.index]
            
            # random feature
            _X = _X.sample(replace = False, 
                           n = self.n_random_features, 
                           axis = 1)
            self.features.append(_X.columns)
            
            # out of bag sample
            self.__oob_index.append(set(X.index) - set(_X.index))
            
            # fit base learner
            _model = _model.fit(_X, _y)
            
            # save the base learner
            self.__models.append(_model)
            
            # calculate variable importance
            if variable_importance:
                self.__variable_importance()
                    
    def predict(self, X):
        predicted_values = []
        
        for _model, _features in zip(self.__models, self.features):
            predicted_values.append(_model.predict(X[_features]))
        
        predicted_values = pd.DataFrame(predicted_values)
        predicted_values = predicted_values.mode().transpose()[0]
        predicted_values.name = 'predicted'
        
        return predicted_values
    
    def predict_proba(self, X):
        predicted_values = np.zeros(shape = (X.shape[0], self.k))
        
        for _model, _features in zip(self.__models, self.features):
            predicted_values += _model.predict_proba(X[_features])
        
        predicted_values /= self.n_models
        predicted_values = pd.DataFrame(predicted_values)
        predicted_values.columns = self.__models[0].classes_
        
        return predicted_values
    
    def __variable_importance(self):
        vi = []
        
        for _model, _oob_index, _features in zip(self.__models, 
                                                 self.__oob_index, 
                                                 self.features):
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
        
class RandomForestRegressor():
    def __init__(self, n_models = 101, n_random_features = None):
        
        self.n_models = n_models
        self.__models = []
        self.__oob_index = []
        self.variable_importance = []
        
        self.features = []
        self.n_random_features = n_random_features
    
    def fit(self, X, y, variable_importance = False):
        self.X = X
        self.y = y
        
        self.n, self.p = X.shape
        
       # mtry
        if self.n_random_features is None:
            self.n_random_features = int(np.sqrt(self.p))
        
        # Training
        for b in range(self.n_models):
            # define base learner
            _model = tree.DecisionTreeRegressor()
            
            # get bootstrap sample
            _X = X.sample(replace = True, n = self.n)
            _y = y.loc[_X.index]
            
            # random feature
            _X = _X.sample(replace = False, 
                           n = self.n_random_features, 
                           axis = 1)
            self.features.append(_X.columns)
            
            # out of bag sample
            self.__oob_index.append(set(X.index) - set(_X.index))
            
            # fit base learner
            _model = _model.fit(_X, _y)
            
            # save the base learner
            self.__models.append(_model)
            
            # calculate variable importance
            if variable_importance:
                self.__variable_importance()
                    
    def predict(self, X):
        predicted_values = []
        
        for _model, _features in zip(self.__models, self.features):
            predicted_values.append(_model.predict(X[_features]))
        
        predicted_values = pd.DataFrame(predicted_values)
        predicted_values = predicted_values.mean()
        predicted_values.name = 'predicted'
        
        return predicted_values
    
    def __variable_importance(self):
        vi = []
        
        for _model, _oob_index, _features in zip(self.__models, 
                                                 self.__oob_index, 
                                                 self.features):
            _oob_X = self.X.loc[_oob_index][_features]
            _oob_y = np.array(self.y.loc[_oob_index]).transpose()
            
            _e_b = np.mean((_model.predict(_oob_X) - _oob_y)**2)
                        
            _d_b = []
            for col in self.X.columns:
                if col in _oob_X.columns:
                    _oob_X_p = _oob_X.copy()
                    _oob_X_p[col] = _oob_X_p[col].sample(frac = 1).values
                    _p = np.mean((_model.predict(_oob_X_p) - _oob_y)**2)
                    _d_b.append(_p - _e_b)
                    
                else:
                    _d_b.append(None)
            
            vi.append(_d_b)
            
        vi = pd.DataFrame(vi)
        vi = vi.mean() / vi.std()
        self.variable_importance = vi.sort_values(ascending = False)