# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:30:04 2024

Test the delta x/delta t approximation of the different ODE systems, in a
thorough way. The expectation is that the approximation should not work well
for systems with a lot of interconnected differential equations.

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

def slugify(value, allow_unicode=False) :
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
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    # results folder
    results_directory = "../local_results/checking-odebench-delta-x"
    
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
            
            # let's plot the trajectory
            fig, ax = plt.subplots(figsize=(10,8))
            for state_variable in state_variables :
                ax.scatter(df_trajectory["t"].values, 
                           df_trajectory[state_variable].values, 
                           alpha=0.7, label=state_variable)
            ax.set_title(str(system["id"]) + ": " + system["eq_description"])
            ax.set_xlabel("t")
            ax.set_ylabel("State variables")
            ax.legend(loc='best')
            fig.tight_layout()
            figure_file_name = "system-%d-trajectory-%d.png" % (system["id"], trajectory_index)
            plt.savefig(os.path.join(system_directory, figure_file_name), dpi=150)
            plt.close(fig)
            
            # now, we need to check that the known ground truth equations for
            # the delta x approximation actually have a good fit
                
if __name__ == "__main__" :
    sys.exit( main() )    
        
