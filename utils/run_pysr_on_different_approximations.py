# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:16:42 2024

The objective of this script is to run PySR on different types of approximations
for the first derivative y'(t) of an ODE system, to then compare the results.

There are two types of approximations, with some variants/hyperparameters:
- F_y (order)
- delta_y/delta_t (smoothing, order)

@author: Alberto
"""
import os

if __name__ == "__main__" :
   
    # hard-coded values 
    # this is the root folder with all the data for all systems
    systems_folder = "../data/odebench/systems"
    # folder with the results
    results_folder = "../local_results/approximation-comparison-sr"
    # the systems that we are actually going to experiment on
    systems_to_run = [1]
    
    # some hyperparameter settings for PySRRegressor
    
    # let's start by listing all subdirectories inside the main folder
    system_folders = [f for f in os.listdir(systems_folder) if f.find("system") != -1]
    print("Found a total of %d folders containing trajectory files!" % len(system_folders))