# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:54:49 2024

This script uses a new environment (from scratch), created because of some issues
with pytorch.

@author: Alberto
"""
import odeformer
from odeformer.model import SymbolicTransformerRegressor

if __name__ == "__main__" :

    dstr = SymbolicTransformerRegressor(from_pretrained=True)
    model_args = {'beam_size':50, 'beam_temperature':0.1}
    dstr.set_model_args(model_args)
    
    