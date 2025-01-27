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
import pandas as pd
import re as regex
import sys

from pysr import PySRRegressor

if __name__ == "__main__" :
   
    # hard-coded values 
    # this is the root folder with all the data for all systems
    #data_folder = "../data/odebench/systems"
    # this folder below has been obtained through another script
    data_folder = "../local_results/check_odebench_all_transformations/"
    # folder where the results will be written to
    results_folder = "../local_results/approximation-comparison-sr-modified"
    # random seed for all random number generators
    random_seed = 42

    # ids of the systems that we are actually going to experiment on;
    # there are a total of 63 different systems
    #systems_to_run = [i for i in range(1, 64)]
    systems_to_run = [55, 59, 61]

    # hyperparameter settings for transformations and experiments
    noise_level = 0.0
    order_Fx = 1
    window_Fx = None
    order_deltax = 4
    window_deltax = None
    
    # TODO some hyperparameter settings for PySRRegressor
    
    # let's start by listing all subdirectories inside the main folder
    system_folders = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    print("Found a total of %d folders containing trajectory files!" % len(system_folders))
    
    # create a list of tuples: system id as an integer, and its path, using regex to extract id
    system_ids_and_folders = [(int(regex.search("([0-9]+)", f).group(1)), f) for f in system_folders]
    # sort it and select only ids included in the experiments
    system_ids_and_folders = [(s_id, s_f) for s_id, s_f in system_ids_and_folders if s_id in systems_to_run]
    system_ids_and_folders = sorted(system_ids_and_folders, key=lambda x : x[0])
    
    print("Experiments will be run on the following systems:", system_ids_and_folders)
    
    # iterate over all selected folders
    for system_id, system_folder in system_ids_and_folders :
        print("Starting experiment on system %d..." % system_id)
        
        # create folder for the results
        system_result_folder = os.path.join(results_folder, "system-%d" % system_id)
        if not os.path.exists(system_result_folder) :
            os.makedirs(system_result_folder)
        
        # find all approximation files
        all_csv_files = [os.path.join(system_folder, f) for f in os.listdir(system_folder)

                                   if f.endswith(".csv")]
        # select only files with the correct hyperparameters
        dydt_approximation_files = [f for f in all_csv_files if f.find("deltax") != -1
                                    and f.find("noise-%.2f" % noise_level) != -1
                                    and f.find("order_%d" % order_deltax) != -1]
        euler_approximation_files = [f for f in all_csv_files if f.find("F_x") != -1
                                     and f.find("noise-%.2f" % noise_level) != -1
                                     and f.find("order_%d" % order_Fx) != -1]
        
        # the part with the window is the hardest to check (regex or awkward ifs?)
        if window_deltax is None :
            dydt_approximation_files = [f for f in dydt_approximation_files
                                        if f.find("smoothed") == -1]
            euler_approximation_files = [f for f in euler_approximation_files
                                        if f.find("smoothed") == -1]
        
        print("dy/dt approximation files:", dydt_approximation_files)
        print("Euler's approximation files:", euler_approximation_files)
        
        # each approximation works slightly differently, so let's treat each
        # group separately
        for dydt_approximation_file in dydt_approximation_files + euler_approximation_files :
            base_file_name = os.path.basename(dydt_approximation_file)
            print("Preparing PySR run on file \"%s\"..." % base_file_name)
            
            df_approximation = pd.read_csv(dydt_approximation_file)
            
            # targets all have recognizable names
            target_columns = [c for c in df_approximation.columns 
                              if (c.endswith("_deltax") or c.startswith("F_"))
                              and not c.endswith("_pred")]
            feature_columns = [c for c in df_approximation.columns 
                               if c not in target_columns
                               and not c.endswith("_pred")]
            
            # iterate over all possible targets
            for target in target_columns :
                result_file_name = os.path.join(system_result_folder, "equations-" + base_file_name[:-4] + "-" + target + ".csv")
                
                # check if the result file already exists, to avoid re-running experiments
                if not os.path.exists(result_file_name) :
                    print("Now running PySR for target \"%s\", %s=f(%s)..." % 
                          (target, target, str(feature_columns)))
                    df_X = df_approximation[feature_columns]
                    df_y = df_approximation[target]
                    
                    sr = PySRRegressor(
                        population_size=50, # for the first experiment, it was 50
                        niterations=1000,
                        batching=False, # use batches instead of the whole dataset
                        model_selection="best",  # Result is mix of simplicity+accuracy
                        binary_operators=["+", "*", "/", "-",],
                        unary_operators=["sin", "cos", "exp", "log", "sqrt",],
                        temp_equation_file=True, # does not clutter directory with temporary files
                        verbosity=1,
                        random_state=random_seed,
                        )
                    
                    sr.fit(df_X, df_y)
                    
                    print("Saving results to file \"%s\"..." % result_file_name)
                    sr.equations_.to_csv(result_file_name, index=False)
                    
                    # it is also interesting to keep track of the 'best' equation
                    # that PySR would normally choose on the Pareto front
                    best_equation_file_name = os.path.join(system_result_folder, "best-equation-" + base_file_name[:-4] + "-" + target + ".tex")
                    with open(best_equation_file_name, "w") as fp :
                        fp.write(sr.latex(precision=10))
                    
                else :
                    print("Results file for target \"%s\" found, skipping to the next..." % target)