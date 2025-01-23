# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:12:14 2025

@author: Alberto
"""
import json
import logging
import numpy as np
import os
import pandas as pd
import re as regex
import sympy
import sys

from sklearn.metrics import r2_score
from sympy.utilities.lambdify import lambdify

# local scripts
from common import close_logging, initialize_logging, slugify
from check_odebench_all_transformations import add_noise, apply_deltax_method, apply_euler_method_smoothing

def main() :
    
    # the objective of this script is to test a few specific deltax and F_x
    # transformations (the most effective for non-noisy data), and see what
    # happens if we tackle trajectories sampled with a lower delta_t
    
    # hard-coded values
    output_folder = "../local_results/variable-deltat-experiments"
    target_systems = [55, 59, 61]
    files_delta_t = {
        '0.06' : '../data/odebench/all_odebench_trajectories.json',
        '0.03' : '../data/odebench/selected_equations_300_points_trajectories.json',
        '0.015' : '../data/odebench/selected_equations_600_points_trajectories.json',
        }
    transformations = {
        'deltax' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 4}},
        'F_x' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'order' : 1}},
        }
    noise_level = 0.05
    random_seed = 42
    
    if noise_level > 0.0 :
        output_folder += "-noise-%.2f" % noise_level
        transformations = {
            'deltax' : {'function' : apply_deltax_method, 
                        'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 15}}},
            'F_x' : {'function' : apply_euler_method_smoothing, 
                     'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 25}}},
            }
    
    # let's start the experiments! first, let's create the folder
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
    
    # then, initialize the logging procedure
    logger = initialize_logging(output_folder, "log", date=False)
    logger.info("Starting experiment, noise=%.2f, random_seed=%d..."
                % (noise_level, random_seed))
    
    # also initialize pseudo-random number generator
    prng = np.random.default_rng(random_seed)
    
    # the data structure which will be used to store the results
    results = {
        'delta_t' : [],
        'system_id' : [],
        'variable' : [],
        'trajectory_id' : [],
        }
    for t in transformations :
        results[t] = []
    
    # iterate over the files
    for key_delta_t, file_delta_t in files_delta_t.items() :
        
        # load the JSON file
        odebench = None
        with open(file_delta_t, "r") as fp :
            odebench = json.load(fp)
        logger.info("Found a total of %d ODE systems for delta_t=%s" % 
                    (len(odebench), key_delta_t))
        
        # double check: are all the systems we are interested in inside the file?
        # then, select all lines in the file which contain those systems
        systems_in_file = [system['id'] for system in odebench if
                           system['id'] in target_systems] 
        odebench = [system for system in odebench if system['id'] in target_systems]
        
        # ok, now we are going to iterate over each system
        for system in odebench :
            # get out all the information we need from the system dictionary
            system_id = system['id']
            state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system['var_description'])
            equations = {state_variables[i] : sympy.sympify(system['substituted'][0][i])
                         for i in range(0, len(state_variables))}
            logger.info("Now working on system #%d, equations %s" % 
                        (system_id, str(equations)))
            
            # and each trajectory, and each technique
            for trajectory_index, trajectory in enumerate(system['solutions'][0]) :
                
                # create a dictionary for the trajectory
                dictionary_trajectory = {}
                dictionary_trajectory['t'] = trajectory['t']
                for v in range(0, len(state_variables)) :
                    dictionary_trajectory[state_variables[v]] = trajectory['y'][v].copy()
                # conditionally add noise
                if noise_level > 0.0 :
                    dictionary_trajectory = add_noise(dictionary_trajectory, noise_level, prng)
                
                df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
                
                # now: apply the transformations, compute the R2 value of the
                # transformed ground truth equation, for each variable
                for state_variable, equation in equations.items() :
                    # start adding values to the results dictionary
                    results['system_id'].append(system_id)
                    results['variable'].append(state_variable)
                    results['trajectory_id'].append(trajectory_index+1)
                    results['delta_t'].append(key_delta_t)
                    
                    for transformation in transformations :
                        # apply the transformation
                        function = transformations[transformation]['function']
                        kwargs = transformations[transformation]['kwargs']
                        df_transformed = function(df_trajectory, **kwargs)
                        
                        # create symbolic equation
                        symbol_delta_t = sympy.Symbol("Delta_t")
                        if transformation.startswith("F_x") :
                            equation = sympy.Mul(symbol_delta_t, equation, evaluate=False)
                            
                        # prepare symbols and values for the lambdified equation
                        equation_symbols = []
                        for c in df_transformed.columns :
                            if c == "delta_t" :
                                equation_symbols.append(sympy.sympify("Delta_t"))
                            else :
                                equation_symbols.append(sympy.sympify(c))
                        symbol_values = [df_transformed[c].values 
                                         for c in df_transformed.columns]
                        
                        # run the lambdified equation and get the values
                        equation_values = lambdify(equation_symbols, equation)(*symbol_values)
                        
                        # compare equation values against transformed ground truth
                        target_column = state_variable + "_" + transformation
                        if transformation.startswith("F_x") :
                            target_column = "F_" + state_variable
                        r2_value = r2_score(df_transformed[target_column].values,
                                            equation_values)
                        
                        # finally, add all the information to the data structure
                        # containing the results
                        results[transformation].append(r2_value)
    
    # debugging
    for k, v in results.items() :
        print("Key \"%s\": %d entries" % (k, len(v)))
    
    # save results to file
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(os.path.join(output_folder, "results.csv"), index=False)
                        
    # close logging
    close_logging(logger)
    logging.shutdown()
    
    return

if __name__ == "__main__" :
    sys.exit( main() )