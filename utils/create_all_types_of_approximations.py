# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:21:59 2024

Utility cript to create all types of data for approximations, to (maybe) be used
later

@author: Alberto
"""
import json
import os
import pandas as pd
import re as regex
import sys

# these two functions are used to compute the finite difference approximation
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference

# using again the same dirty, deprecated trick to import the local script which
# creates the approximation based on the forward Euler's method
sys.path.append("../src")
from explicit_euler_method import apply_euler_method

if __name__ == "__main__" :
    
    # hard-coded values
    odebench_json_file_name = "../data/odebench/all_odebench_trajectories.json"
    results_folder = "../data/odebench/systems"
    # these are basically hyperparameter values used for hyperparameter optimization
    # in the ODEFormer/ODEBench paper
    finite_difference_orders = [2, 3, 4]
    smoother_window_lengths = [None, 15]
    
    # read JSON file
    print("Reading file \"%s\", containing the trajectories..." % odebench_json_file_name)
    odebench = None
    with open(odebench_json_file_name, "r") as fp :
        odebench = json.load(fp)
    
    # iterate over all systems
    for system in odebench :
        print("Now working on system %d (\"%s\")" % (system["id"], system["eq_description"]))
        
        # check if the system folder already exists, and if it does not, create it
        system_folder = os.path.join(results_folder, "system-%d" % system["id"])
        if not os.path.exists(system_folder) :
            os.makedirs(system_folder)
            
        # get the names of the state variables
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        print("The system has state variables:", state_variables)
            
        # for each trajectory
        trajectories = system["solutions"][0]
        for trajectory_index, trajectory in enumerate(trajectories) :
            
            # first, create a nice pandas DataFrame for the trajectory
            print("Writing trajectory %d..." % trajectory_index)
            dictionary_trajectory = {v : [] for v in ["t"] + state_variables}
            dictionary_trajectory["t"] = trajectory["t"]
            for i in range(0, len(state_variables)) :
                dictionary_trajectory[state_variables[i]] = trajectory["y"][i]
            
            df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
            df_trajectory.to_csv(os.path.join(system_folder, "trajectory-%d.csv" % trajectory_index), index=False)
            
            # then, apply the two data transformations for the approximations;
            # first, the function from SINDy that generates the delta_y/delta_t
            # with all the possible values used in the ODEFormer paper
            for finite_difference_order in finite_difference_orders :
                for smoother_window_length in smoother_window_lengths :
                    
                    # generate an instance of an object able to manage the differentiation,
                    # depending on the settings; in the meantime, prepare file name
                    df_dy_dt_file_name = "trajectory-%d-order%d" % (trajectory_index, finite_difference_order)
                    fd = None
                    if smoother_window_length is None :
                        fd = FiniteDifference(order=finite_difference_order)
                    else :
                        fd = SmoothedFiniteDifference(order=finite_difference_order,
                                                      smoother_kws={'window_length': smoother_window_length}
                                                      )
                        df_dy_dt_file_name += "-window%d" % smoother_window_length
                    dy_dt = fd._differentiate(df_trajectory[state_variables].values, 
                                      df_trajectory["t"].values)
                    
                    # prepare another pandas DataFrame
                    dy_dt_column_names = [v + "_dxdt" for v in state_variables]
                    # most of it will be a copy of the previous one
                    df_dy_dt = df_trajectory.copy()
                    for i in range(0, len(state_variables)) :
                        df_dy_dt[dy_dt_column_names[i]] = dy_dt[:,i]
                    
                    # also save it to disk
                    df_dy_dt.to_csv(os.path.join(system_folder, df_dy_dt_file_name + ".csv"), index=False)
            
            # and then, apply the forward Euler's method approximation to get F    
            df_euler = apply_euler_method(df_trajectory)
            df_euler.to_csv(os.path.join(system_folder, "trajectory-%d-euler.csv" % trajectory_index), index=False)
