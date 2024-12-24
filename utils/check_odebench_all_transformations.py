# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:00:06 2024

Checking what should be the 'correct' equations of an ODE system against the
different data transformations. The flow of the program is:
1. load odebench
2. for each ODE system in odebench, obtain ground truth equations for each transformation
3. for each trajectory, for each variable:
    3a. compute data transformation for $delta x$, check fitting w.r.t. ground truth
    3b. compute data transformation for $F_x$, check fitting w.r.t. ground truth
4. repeat 3, but for different levels of noise

@author: Alberto
"""
import json
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import re as regex
import seaborn as sns
import sympy
import sys

from sklearn.metrics import r2_score
from sympy.utilities.lambdify import lambdify

# local scripts
from common import close_logging, initialize_logging, slugify

def add_noise(dictionary_trajectory, noise_level, prng) :
    """
    Add random Gaussian/white noise.
    """
    # the expected input is a dictionary with keys 't' + state variables
    state_variables = [k for k in dictionary_trajectory if k != 't']
    
    for v in state_variables :
        trajectory = np.array(dictionary_trajectory[v])
        noise = prng.normal(loc=0.0, scale=noise_level * np.abs(trajectory))
        dictionary_trajectory[v] = trajectory + noise
        
    return dictionary_trajectory

def plot_trajectory(system_name, dictionary_trajectory, trajectory_file_name) :
    """
    Plot trajectory for state variables given in dictionary.
    """
    t = dictionary_trajectory['t']
    state_variables = [k for k in dictionary_trajectory if k != 't']
    
    fig, ax = plt.subplots()
    for v in state_variables :
        ax.plot(t, dictionary_trajectory[v], label=v)
    ax.set_xlabel('t')
    ax.set_ylabel('State variables')
    ax.set_title(system_name + " - " + os.path.basename(trajectory_file_name))
    ax.legend(loc='best')
    
    plt.savefig(trajectory_file_name, dpi=150)
    plt.close(fig)
    
    return

def main() :
    # some hard-coded values
    # source file, in JSON
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    # results folder
    results_directory = "../local_results/" + os.path.basename(__file__)[:-3]
    # noise levels (check ODEFormer paper)
    noise_levels = [0.0, 0.01]
    # we also need a pseudo-random number generator
    random_seed = 42
    prng = np.random.default_rng(random_seed)
    # set a nice style for plots
    sns.set_style('darkgrid')
    
    # let's also perform some proper logging
    logger = initialize_logging(results_directory, "log", date=False)
    
    # load odebench
    logger.info("Loading ODEBench file \"%s\"..." % odebench_file_name)
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
    logger.info("Found a total of %d ODE systems" % len(odebench))
    
    # start iterating over systems
    for system in odebench :
        logger.info("Working on system %d: \"%s\"" % (system["id"], system["eq_description"]))
        
        # let's create a subdirectory for this system
        system_directory = slugify("%d-%s" % (system["id"], system["eq_description"]))
        system_directory = os.path.join(results_directory, system_directory)
        if not os.path.exists(system_directory) :
            os.makedirs(system_directory)
        logger.info("Results will be saved in folder \"%s\"" % system_directory)
        
        # get the description of the state variables, and capture their names
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        # associate each variable with the expression of its derivative; for
        # some weird reason, the key "substituted" returns a list of ONE element (!)
        # that contains a list of strings ^_^;;
        equations = {state_variables[i] : sympy.sympify(system["substituted"][0][i])
                     for i in range(0, len(state_variables))}
        logger.info("System equations: %s" % str(equations))
        
        # get the trajectories
        # now, we can proceed to read the data (integrated trajectory/ies); there
        # are in fact TWO distinct trajectories with different initial conditions (!)
        trajectories = system["solutions"][0] # for some reason, another list with 1 element (...) 
        logger.info("Found a total of %d trajectories!" % len(trajectories))
        
        # iterate over: trajectories, levels of noise
        for trajectory_index, trajectory in enumerate(trajectories) :
            for noise_level in noise_levels :
                logger.info("Working on trajectory %d, noise level %.2f" %
                            (trajectory_index, noise_level))
                # now, the weird thing about the data structure is that values for
                # 't' are presented as a key in the dictionary; while all the others
                # are presented as items in a list under key 'y' (...)
                dictionary_trajectory = {}
                dictionary_trajectory["t"] = trajectory["t"]
                
                for v in range(0, len(state_variables)) :
                    dictionary_trajectory[state_variables[v]] = trajectory["y"][v].copy()
                
                # if noise level is not zero, generate a new trajectory
                if noise_level > 0.0 :
                    dictionary_trajectory = add_noise(dictionary_trajectory, noise_level, prng)
                
                # save file with trajectory data
                trajectory_file_name = os.path.join(system_directory, 
                                                    "trajectory-%d-noise-%.2f" % (trajectory_index, noise_level))
                df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
                df_trajectory.to_csv(trajectory_file_name + ".csv", index=False)
                
                # also plot the trajectorys
                plot_trajectory(os.path.basename(system_directory),
                                dictionary_trajectory, 
                                trajectory_file_name + ".png")
                
                
    # close log
    close_logging(logger)
    logging.shutdown()

if __name__ == "__main__" :
    sys.exit( main() )
