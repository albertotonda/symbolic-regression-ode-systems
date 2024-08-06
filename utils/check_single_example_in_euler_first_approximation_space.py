# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:40:49 2024

Maybe redundant, but this is just a script to manipulate and perform some checks
on specific benchmarks from the ODEBench benchmark suite.

@author: Alberto
"""
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import re as regex
import seaborn as sns
import sympy
import sys
import unicodedata # this is just used to sanitize strings for file names

from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sympy.utilities.lambdify import lambdify

# local package(s), imported from another directory
# using a 'parent package' would be cleaner; but for the moment, let's use a brutal way
sys.path.append("../src/")
from explicit_euler_method import apply_euler_method

# this function should probably be moved in a more shared script
def median_r2_score(y_true, y_pred) :
    
    y_mean = np.mean(y_true)
    numerator = (np.median(abs(y_true - y_pred), axis=0))**2
    denominator = np.sum((y_true - y_mean)**2, axis=0)
    
    return 1.0 - (numerator/denominator)

def main() :
    
    # hard-coded values
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    #odebench_file_name = "../data/odebench/selected_equations_300_points_trajectories.json"
    # specify id of the system we want to analyze; the worst results are with systems
    # 11, 11, 26, 41, 48, 49, 54, 55, 56, 57, 58, 59 and 61
    system_id = 11
    # also select one of the two trajectories in the system, trajectory 0 or 1
    trajectory_index = 1
    # this is optional, it's a dictionary of alternative candidate for F_y 
    # besides the ground truth, one for each state variable
    #alternative_equations = {'x_0' : 'Delta_t*(-x_0 + sin(x_0))*(sqrt(sin(sin(exp((-Delta_t + x_0)*(-x_0 - 0.1215392))))) + 5.1706514)'}
    #alternative_equations = {'x_0' : '1.2608093 * sin(sin(Delta_t / (1.5884168 / x_0)) / 1.1236044)'}
    #alternative_equations = {'x_0' : '((x_0 * -0.35642046) + 0.2993954) * Delta_t'}
    alternative_equations = {'x_0' : 'Delta_t * (-5.201244*x_0 + 5.14734730533612 * sin(x_0))'}
    
    # set a nice style for plots
    sns.set_style('darkgrid')
    
    # read the file with all the trajectories
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
        
    # select the system and start working on it
    i = 0
    while odebench[i]["id"] != system_id and i < len(odebench) :
        i += 1
    system = odebench[i]
    
    print("Working on system %d: \"%s\"" % (system["id"], system["eq_description"]))
    state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
    # associate each variable with the expression of its derivative; for
    # some weird reason, the key "substituted" returns a list of ONE element (!)
    # that contains a list of strings ^_^;;
    equations = {state_variables[i] : sympy.sympify(system["substituted"][0][i])
                 for i in range(0, len(state_variables))}
    
    # now, we can proceed to read the data (integrated trajectory/ies); there
    # are in fact TWO distinct trajectories with different initial conditions (!)
    trajectories = system["solutions"][0] # for some reason, another list with 1 element (...) 
    print("Found a total of %d trajectories, selected trajectory %d!" % 
          (len(trajectories), trajectory_index))
    trajectory = trajectories[trajectory_index]
    
    # transform the information on the trajectory into a DataFrame, then call
    # the function to convert it into the forward Euler's method's space
    print("Transforming trajectory into the forward Euler's method's space...")
    dictionary_trajectory = {}
    dictionary_trajectory["t"] = trajectory["t"]
    for v in range(0, len(state_variables)) :
        dictionary_trajectory[state_variables[v]] = trajectory["y"][v]
    df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
    df_euler = apply_euler_method(df_trajectory)
    
    # now, check what happens for each state variable
    for state_variable, equation in equations.items() :
        print("Now working on state variable \"%s\":" % state_variable)
        delta_t = sympy.Symbol("Delta_t")
        euler_equation = sympy.Mul(delta_t, equation, evaluate=False)
        print("Ground truth equation:", euler_equation)
        
        # prepare symbols and values for the lambdified equation
        equation_symbols = []
        for c in df_euler.columns :
            if c == "delta_t" :
                equation_symbols.append(sympy.sympify("Delta_t"))
            else :
                equation_symbols.append(sympy.sympify(c))
        symbol_values = [df_euler[c].values for c in df_euler.columns]
        
        # obtain predicted values
        equation_values = lambdify(equation_symbols, euler_equation)(*symbol_values)
        ground_truth = df_euler["F_" + state_variable].values
        
        # compute R2 score
        r2_value = r2_score(ground_truth, equation_values)
        median_r2_value = median_r2_score(ground_truth, equation_values)
        mse_value = mean_squared_error(ground_truth, equation_values)
        median_error_value = median_absolute_error(ground_truth, equation_values)
        
        print("- Ground truth equation for \"%s\": R2=%.6f" % (state_variable, r2_value))
        print("- Ground truth equation for \"%s\": MSE=%.6e" % (state_variable, mse_value))
        print("- Ground truth equation for \"%s\": MedR2=%.6f" % (state_variable, median_r2_value))
        print("- Ground truth equation for \"%s\": MedAE=%.6e" % (state_variable, median_error_value))
        
        # check if an alternative candidate has been specified for this state variable
        if state_variable in alternative_equations :
            alternative_equation = sympy.sympify(alternative_equations[state_variable])
            alternative_values = lambdify(equation_symbols, alternative_equation)(*symbol_values)
        
            r2_value = r2_score(ground_truth, alternative_values)
            median_r2_value = median_r2_score(ground_truth, alternative_values)
            mse_value = mean_squared_error(ground_truth, alternative_values)
            median_error_value = median_absolute_error(ground_truth, alternative_values)
            
            print("Alternative equation:", alternative_equation)
            print("- Alternative equation for \"%s\": R2=%.6f" % (state_variable, r2_value))
            print("- Alternative equation for \"%s\": MSE=%.6e" % (state_variable, mse_value))
            print("- Alternative equation for \"%s\": MedR2=%.6f" % (state_variable, median_r2_value))
            print("- Alternative equation for \"%s\": MedAE=%.6e" % (state_variable, median_error_value))
            
    
    return

if __name__ == "__main__" :
    sys.exit(main())
