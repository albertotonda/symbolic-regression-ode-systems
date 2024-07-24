# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:23:11 2024

Similar to the script "check_equations_in_euler_first_approximation_space.py",
but this time we are going to read a file created by the ODEBench benchmark
and proceed from that.

@author: Alberto
"""
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re as regex
import seaborn as sns
import sympy
import sys
import unicodedata # this is just used to sanitize strings for file names

from sklearn.metrics import r2_score
from sympy.utilities.lambdify import lambdify

# local package(s)
from explicit_euler_method import apply_euler_method

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = regex.sub(r'[^\w\s-]', '', value.lower())
    return regex.sub(r'[-\s]+', '-', value).strip('-_')

def main() :
    
    # some hard-coded values
    # source file, in JSON
    odebench_file_name = "../local_files/odeformer/odeformer/odebench/solutions.json"
    # this one is another file that allegedly should contain the same information,
    # but it is in fact different (!)
    #odebench_file_name = "../local_files/odeformer/odeformer/odebench/strogatz_extended.json"
    # results folder
    results_directory = "../local_results/checking-odebench"
    
    # this data structure will be used to collect the results
    results = []
    
    # set a nice style for plots
    sns.set_style('darkgrid')
    
    if not os.path.exists(results_directory) :
        os.makedirs(results_directory)
    
    # read the file
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
        
    # for some reason, the JSON file is a list of dictionaries, so let's go with the flow
    for system in odebench :
        print("Working on system %d: \"%s\"" % (system["id"], system["eq_description"]))
        
        # let's create a subdirectory for this system
        system_directory = slugify("%d-%s" % (system["id"], system["eq_description"]))
        system_directory = os.path.join(results_directory, system_directory)
        if not os.path.exists(system_directory) :
            os.makedirs(system_directory)
        
        # get the description of the state variables, and capture their names
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        #print("State variables:", state_variables)
        
        # start preparing the sub-dictionary for the results
        results_system = {s : {} for s in state_variables}
        results_system["id"] = system["id"]
        results_system["eq_description"] = system["eq_description"]
        
        # associate each variable with the expression of its derivative; for
        # some weird reason, the key "substituted" returns a list of ONE element (!)
        # that contains a list of strings ^_^;;
        equations = {state_variables[i] : sympy.sympify(system["substituted"][0][i])
                     for i in range(0, len(state_variables))}
        
        # now, we can proceed to read the data (integrated trajectory/ies); there
        # are in fact TWO distinct trajectories with different initial conditions (!)
        trajectories = system["solutions"][0] # for some reason, another list with 1 element (...) 
        print("Found a total of %d trajectories!" % len(trajectories))
        
        # let's go trajectory by trajectory
        for trajectory_index, trajectory in enumerate(trajectories) :
            
            # now, the weird thing about the data structure is that values for
            # 't' are presented as a key in the dictionary; while all the others
            # are presented as items in a list under key 'y' (...)
            dictionary_trajectory = {}
            dictionary_trajectory["t"] = trajectory["t"]
            
            for v in range(0, len(state_variables)) :
                dictionary_trajectory[state_variables[v]] = trajectory["y"][v]
            
            print("- Trajectory #%d contains %d points" % (trajectory_index, len(dictionary_trajectory["t"])))
        
            # convert the dictionary with the values to a pandas DataFrame,
            # and save it (why not?)
            df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
            df_trajectory.to_csv(os.path.join(system_directory, "trajectory-%d.csv" % trajectory_index), index=False)
            
            # let's plot the trajectories!
            for state_variable in state_variables :
                
                fig, ax = plt.subplots(figsize=(10,8))
                ax.scatter(df_trajectory["t"].values, df_trajectory[state_variable].values, alpha=0.7)
                ax.set_title("Trajectory %d, state variable \"$%s$\"" % (trajectory_index, state_variable))
                ax.set_xlabel("t")
                ax.set_ylabel("$" + state_variable + "$")
                fig.tight_layout()
                
                figure_file_name = "trajectory-%d-%s.png" % (trajectory_index, state_variable)
                plt.savefig(os.path.join(system_directory, figure_file_name), dpi=150)
                plt.close(fig)
                
            # now, we need to check that the known ground truth equations for
            # the F_y functions actually have a good fit; let's first transform
            # the data set into the forward Euler's method feature space
            df_euler = apply_euler_method(df_trajectory)
            df_euler.to_csv(os.path.join(system_directory, "trajectory-%d-euler.csv" % trajectory_index), index=False)
            
            # for each state variable y, obtain the F_y known ground truth form
            for state_variable, equation in equations.items() :
                delta_t = sympy.Symbol("delta_t")
                euler_equation = sympy.Mul(delta_t, equation)
                
                # prepare symbols and values for the lambdified equation
                equation_symbols = [sympy.sympify(s) for s in df_euler.columns]
                symbol_values = [df_euler[c].values for c in df_euler.columns]
                
                # obtain predicted values
                equation_values = lambdify(equation_symbols, euler_equation)(*symbol_values)
                ground_truth = df_euler["F_" + state_variable].values
                
                # compute R2 score
                r2_value = r2_score(ground_truth, equation_values)
                
                # TODO save results in a better format?
                print("-- For state variable \"%s\", real F_%s has R2=%.6f" %
                      (state_variable, state_variable, r2_value))
                
                # save results in the dictionary
                results_system[state_variable]["F"] = str(euler_equation)
                results_system[state_variable]["trajectory-%d" % trajectory_index] = r2_value
                
        #sys.exit(0) # TODO remove this, it's just for debugging
        # here is the end of the loop for a system
        results.append(results_system)
        
    # TODO it would be nice to create a Latex table starting from the results
    # collected in the "results" list; since there are 63 (!) systems, maybe
    # I could copy what the people in ODEBench did, and just go for separate
    # tables, depending on the number of equations in the system
    
    return

if __name__ == "__main__" :
    sys.exit(main())