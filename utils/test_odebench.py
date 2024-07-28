# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:26:02 2024

Not super-clean, but let's try a new version of the full pipeline on the examples
of the ODEBench.

@author: Alberto
"""
import datetime
import itertools
import json
import juliacall
import os
import pandas as pd
import pickle
import re as regex
import shutil
import sys

from sklearn.metrics import mean_squared_error, r2_score

from pysr import PySRRegressor
from scipy import integrate
from sympy.parsing.sympy_parser import parse_expr

# local libraries; in order to include I need to use this trick with 'sys' that is
# not super-clean...
sys.path.append("../src/")
from create_dataset_from_ode_system import get_df_from_ode_system, parse_ode_from_text
from explicit_euler_method import apply_euler_method
from learn_equations import prune_equations
from optimize_ode_systems import dX_dt
from local_utility import MyTimeoutError, run_process_with_timeout

# this function could also be moved to another, more general, script
# it's a string reporting the Julia loss function implementing median absolute error
loss_function_julia_string = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    
    absolute_errors = abs.(prediction .- dataset.y)
    
    sorted_absolute_errors = sort(absolute_errors)
    n = length(sorted_absolute_errors)
    
    median_absolute_error = 0.0
    if n % 2 == 1
        median_absolute_error = sorted_absolute_errors[(n + 1) ÷ 2]
    else
        median_absolute_error = (sorted_absolute_errors[n ÷ 2] + sorted_absolute_errors[n ÷ 2 + 1]) / 2
    end
    
    mean_squared_error = sum((prediction .- dataset.y) .^ 2) / dataset.n
    
    return 0.5 * mean_squared_error + 0.5 * median_absolute_error
    
end
"""

if __name__ == "__main__" :
    
    # hard-coded values
    random_seed = 42
    timeout = 120 # in seconds
    results_folder = "../local_results/results-odebench"
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    
    # we could iterate over all benchmarks in ODEBench, but it's more likely
    # that we will just go for one, selected by its id
    selected_system_ids = [system_id for system_id in range(1, 24)]
    
    # read the file containing all the trajectories
    print("Reading file \"%s\", containing the trajectories..." % odebench_file_name)
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
        
    # if no system id has been specified, we are going to work on all of them
    if len(selected_system_ids) == 0 :
        selected_system_ids = [system["id"] for system in odebench]
    
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    
    # iterate over all benchmarks in ODEBench, but only if no system_id has been specified
    print("I am going to work on systems:", selected_system_ids)
    odebench = [system for system in odebench if system["id"] in selected_system_ids]
    for system in odebench :
        
        print("Now working on system #%d, \"%s\"..." % (system["id"], system["eq_description"]))
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        print("The system has state variables:", state_variables)
        # create sub-folder for the system's results
        system_folder = os.path.join(results_folder, "system-%d" % system["id"])
        if not os.path.exists(system_folder) :
            os.makedirs(system_folder)
            
        # we can also save the ground truth of the system as a file, it might be
        # useful later to perform comparisons
        ground_truth_system_structure_file_name = os.path.join(system_folder, "ground-truth-system.txt")
        if not os.path.exists(ground_truth_system_structure_file_name) :
            
            with open(ground_truth_system_structure_file_name, "w") as fp :
                for i in range(0, len(system["substituted"][0])) : # again, another list of one element...
                    fp.write("d" + state_variables[i] + "/dt = " + system["substituted"][0][i] + "\n")
        
        # read the information from the ODEBench trajectories, and save them to
        # corresponding dataframe files; trajectories are stored inside a list
        # of ONE element (...) under the key 'solutions' of the system dictionary
        # we also need to store all DataFrame with forward Euler's method's space
        df_euler_list = []
        for trajectory_index, trajectory in enumerate(system["solutions"][0]) :
            dictionary_trajectory = {}
            dictionary_trajectory["t"] = trajectory["t"]
            for v in range(0, len(state_variables)) :
                dictionary_trajectory[state_variables[v]] = trajectory["y"][v]
            
            # TODO split data? cross-validation? this is the point to do it
            
            # save the two files
            df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
            df_trajectory.to_csv(os.path.join(system_folder, "trajectory-%d.csv" % trajectory_index), index=False)
            df_euler = apply_euler_method(df_trajectory)
            df_euler.to_csv(os.path.join(system_folder, "trajectory-%d-euler.csv" % trajectory_index), index=False)
            
            # store the DataFrame in the list
            df_euler_list.append(df_euler)
            
        # now, in order to create the data that will be used for training the
        # Symbolic Regression algorithm, we need to concatenate the DataFrames
        df_euler = pd.concat(df_euler_list)
        
        # prepare some data structures; they are necessary, so that if we already
        # have the results for one of the state variables, we do not re-run the algorithm
        target_names = [c for c in df_euler.columns if c.startswith("F_")]
        dictionary_regressors = {} # used to store the full regressor objects
        dictionary_equations = {} # used to store the equations in the F_y feature space
        dictionary_pruned_equations = {} # used to store the equations in the original feature space
        
        for target in target_names :
            target_equations_file_name = os.path.join(system_folder, "equations-%s.csv" % target)
            target_regressor_file_name = os.path.join(system_folder, "regressor-%s.pkl" % target)
            
            if not os.path.exists(target_equations_file_name) or not os.path.exists(target_regressor_file_name) :
                print("Now running symbolic regression for variable \"%s\"..." % target)
                
                # create new DataFrames, selecting the columns for features and target
                X = df_euler[[c for c in df_euler.columns if c != target and not c.startswith("F_")]]
                y = df_euler[target]
                
                # initialize PySRRegressor
                symbolic_regressor = PySRRegressor(
                    population_size=50, # for the first experiment, it was 50
                    niterations=1000,
                    batching=False, # use batches instead of the whole dataset
                    batch_size=50, # 50 is the default value for the batches
                    model_selection="best",  # Result is mix of simplicity+accuracy
                    binary_operators=["+", "*", "/", "-", "pow(x,y)=x^y"],
                    unary_operators=["sin", "cos", "exp", "log", "sqrt",],
                    extra_sympy_mappings={"pow" : lambda x, y : x**y},
                    #loss_function=loss_function_julia_string,
                    #early_stop_condition=("stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"), # stop early if we find a good and simple equation
                    temp_equation_file=True, # does not clutter directory with temporary files
                    verbosity=1,
                    random_state=random_seed,
                    )
                
                # finally, run the Symbolic Regression!
                symbolic_regressor.fit(X, y)
                
                # save Pareto front of equations to file, mostly for debugging
                symbolic_regressor.equations.to_csv(target_equations_file_name, index=False)
                dictionary_equations[target] = symbolic_regressor.equations
                
                # also store the regressor objects in the dictionary and save them as pickles
                dictionary_regressors[target] = symbolic_regressor
                with open(target_regressor_file_name, "wb") as fp :
                   pickle.dump(symbolic_regressor, fp)
        
            else :
                print("File with equations for target \"%s\" found, reading it and skipping to the next step..."
                      % target)
                dictionary_equations[target] = pd.read_csv(target_equations_file_name)
                with open(target_regressor_file_name, "rb") as fp :
                   dictionary_regressors[target] = pickle.load(fp)
                         
            # next step: apply the methodology to go back to the original feature space
            # which also prunes equations reduced to '0' or with impossible values
            variable_name = target[2:] # target is 'F_y' with 'y' being the name of the variable
            pruned_equations_file_name = os.path.join(system_folder, "equations-pruned-%s.csv" % variable_name)
            
            if not os.path.exists(pruned_equations_file_name) :
                print("Pruning equations for target \"%s\" -> \"%s\"..." % (target, variable_name))
                # take the equations in the data frame obtained by the symbolic regressor,
                # and turn them into symbolic expressions, using sympy parse_expr
                equations = [ parse_expr(eq) for eq in dictionary_equations[target]["equation"].values ]
                pruned_equations = prune_equations(equations)
                
                # save the pruned equations as a CSV
                df_pruned_equations = pd.DataFrame.from_dict({"equation" : pruned_equations})
                df_pruned_equations.to_csv(pruned_equations_file_name, index=False)
                
                # also store them in the dictionary, using the original variable name
                dictionary_pruned_equations[variable_name] = pruned_equations
                
            else :
                print("Found file with pruned equations for target \"%s\", reading it and skipping to the next step..."
                      % target)
                df_pruned_equations = pd.read_csv(pruned_equations_file_name)
                dictionary_pruned_equations[variable_name] = df_pruned_equations["equation"].values.tolist()