# -*- coding: utf-8 -*-

import numpy as np

class Preprocessor:
    
    def standardize(dataset):
        
        for column in dataset[:-1].columns:
            col_mean = np.mean(dataset[column])
            col_std = np.std(dataset[column])
            
            dataset[column] = (dataset[column] - col_mean) / col_std
        
        return dataset
    
    def normalize(dataset):
        
        for column in dataset[:-1].columns:
            col_min = np.min(dataset[column])
            col_max = np.max(dataset[column])
            
            dataset[column] = (dataset[column] - col_min) / (col_max-col_min)
        
        return dataset
    
    
    