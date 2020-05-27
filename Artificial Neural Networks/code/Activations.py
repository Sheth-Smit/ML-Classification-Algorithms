# -*- coding: utf-8 -*-

import numpy as np

class Activations:
    
    def activation(x, act_type='sigmoid'):
        
        if act_type == 'sigmoid':
            return (1 / (1 + np.exp(-x)))
        elif act_type == 'tanh':
            return np.tanh(x)
        elif act_type == 'linear':
            return x
        elif act_type == 'relu':
            return np.array([max(num, 0) for num in x])
    
    def deriv_activation(x, act_type='sigmoid'):
        
        if act_type == 'sigmoid':
            sig = Activations.activation(x, act_type='sigmoid')
            return sig * (1-sig)
        elif act_type == 'tanh':
            z = np.tanh(x)
            return 1-(z**2)
        elif act_type == 'linear':
            return 1
        elif act_type == 'relu':
            deriv = []
            for num in x:
                if num != 0:
                    deriv.append(1)
                else:
                    deriv.append(0)
            
            return np.array(deriv)
