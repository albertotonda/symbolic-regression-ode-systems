# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:56:01 2024

Collect all data produced by an experimental run, and organize it (hopefully
neatly) inside a table.

@author: Alberto
"""
import json
import os
import pandas as pd
import re as regex
import sys

from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__" :
    
    # odebench source file, in JSON
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    experiment_folder = "../local_results/check_odebench_all_transformations"
    output_file_name = os.path.join(experiment_folder, "summary.csv")
    
    metric_function = r2_score
    #metric_function = mean_squared_error
    
    # load odebench
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
    
    # data structure to collect the results
    results = {'system_id' : [], 'trajectory_id' : [], 'variable' : []}
    
    # start iterating over systems
    for system in odebench :
        system_id = system['id']
        print("Now working on system %d..." % system_id)
        
        # select all folders that start with that
        system_folder = [f for f in os.listdir(experiment_folder) if f.startswith(str(system_id))]
        system_folder = os.path.join(experiment_folder, system_folder[0])
        print("Folder: \"%s\"" % system_folder)
        
        # get all files that end with .csv from inside the folder
        csv_files = [os.path.join(system_folder, f) for f in os.listdir(system_folder)
                     if f.endswith(".csv")]
        
        # get some information out of the files and organize a dictionary of
        # DataFrames
        dfs = {}
        for csv_file in csv_files :
            m = regex.match("trajectory\-([0-9]+)\-noise\-([0-9|\.]+)\_([\w]+)\.csv",
                            os.path.basename(csv_file))
            if m is not None :
                data_transformation = m.group(3)
                trajectory_id = int(m.group(1))
                noise_level = float(m.group(2))
                print("Found file for trajectory %d, noise %.2f, data transformation \"%s\"" % 
                      (trajectory_id, noise_level, data_transformation))
                
                # prepare the column
                column_name = data_transformation + "_%.2f" % noise_level
                if column_name not in results :
                    results[column_name] = []
                
                # load dataframe, identify columns related to state variables
                df = pd.read_csv(csv_file)
                
                ground_truth_variables = [c for c in df.columns 
                                          if c.find("pred") == -1 
                                          and c.find("_x_hat") == -1 
                                          and c != 't'
                                          and c != 'delta_t'
                                          and (c.find("deltax") != -1 or c.find("F_x") != -1)]
                     
                # for each ground truth variable, compute R2 and store a row
                # inside the results dictionary
                for v in ground_truth_variables :
                    # ground truth values
                    ground_truth = df[v].values
                    # now, the predicted values in the transformed space can have
                    # two different names; either with _pred, or with _x_hat
                    key_prediction = v + "_pred"
                    if key_prediction not in df.columns :
                        key_prediction = v + "_x_hat"
                        if key_prediction not in df.columns :
                            print("Error! Key for predicted values not found!")
                            sys.exit(0)
                    
                    predicted = df[key_prediction].values
                    metric_value = metric_function(ground_truth, predicted)
                    results[column_name].append(metric_value)
                    
                    original_variable = v[2:]
                    if v.find('deltax') != -1 :
                        original_variable = v[:3]
                    
                    if(len(results[column_name]) > len(results['system_id'])) :
                        results['system_id'].append(system_id)
                        results['trajectory_id'].append(trajectory_id)
                        results['variable'].append(original_variable)
    
    # due to differences in the transformations, some of the keys might be empty
    # so we should prune them
    print("Pruning empty keys...")
    keys_to_be_removed = []
    for key, value in results.items() :
        print("Key \"%s\" has length %d" % (key, len(value)))
        
        if len(value) == 0 :
            keys_to_be_removed.append(key)
            
    for key in keys_to_be_removed :
        results.pop(key, None)
        
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(output_file_name, index=False)