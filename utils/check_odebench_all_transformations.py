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

from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference

# these two lines are for denoising, but I need to think more on how to apply it
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from scipy.signal import savgol_filter # used to smoothen data

from sympy.utilities.lambdify import lambdify

# local scripts
from common import close_logging, initialize_logging, slugify

# local package(s), imported from another directory
# using a 'parent package' would be cleaner; but for the moment, let's use a brutal way
sys.path.append("../src/")
from explicit_euler_method import apply_improved_euler_method

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

def apply_deltax_method(df_trajectory, order=2, smoothing=None, denoising=None) :
    """
    This is just a wrapper for pysindy's method.
    """
    # first, let's extract the necessary information from the DataFrame
    state_variables = [c for c in df_trajectory.columns if c != 't']
    X = df_trajectory[state_variables].values
    t = df_trajectory['t'].values
    
    # instantiate pysindy's object for differentiation
    fd = None
    if smoothing is None :
        fd = FiniteDifference(order=order)
    else :
        fd = SmoothedFiniteDifference(order=order, smoother_kws=smoothing)
        
    # obtain numpy array
    dx_dt = fd._differentiate(X, t)
    
    # re-transform everything to a DataFrame
    dictionary_deltax = df_trajectory.to_dict(orient='list')
    for i in range(0, dx_dt.shape[1]) :
        dictionary_deltax[state_variables[i] + "_deltax"] = dx_dt[:,i]
    
    df_deltax = pd.DataFrame.from_dict(dictionary_deltax)
    
    return df_deltax

def apply_euler_method_smoothing(df_trajectory, order=1, smoothing=None) :
    """
    This is just a wrapper around Euler's method. However, we can also add some
    smoothing, using the same functions as pysindy's.
    """
    if smoothing is not None :
        # extract the values that need to be smoothed
        df_smoothed = df_trajectory.copy()
        features = [c for c in df_smoothed.columns if c != 't']
        X = df_smoothed[features].values
        
        # use pysindy's stuff to manage the smoothing
        smoothing["polyorder"] = 3
        smoothing["axis"] = 0
        X_smoothed = savgol_filter(X, **smoothing)
        
        # rearrange the dataframe
        df_smoothed[features] = X_smoothed
        df_trajectory = df_smoothed
    
    df_euler = apply_improved_euler_method(df_trajectory, delta_t=order)
    
    return df_euler

def compare_data_transformations(equations, dictionary_trajectory, trajectory_name) :
    """
    Compute 'ground truth' form of the equations using the different transformations,
    and compare them against the transformed data
    """
    # this is a dictionary of data transformations; key -> function, args, kwargs
    data_transformations = {
        'F_x_deltat_1' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 1}},
        'F_x_deltat_2' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 2}},
        'F_x_deltat_3' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 3}},
        
        'F_x_deltat_1_smoothing' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 1, 'smoothing' : {'window_length' : 15}}},
        'F_x_deltat_2_smoothing' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 2, 'smoothing' : {'window_length' : 15}}},
        'F_x_deltat_3_smoothing' : {'function' : apply_euler_method_smoothing, 'kwargs' : {'delta_t' : 3, 'smoothing' : {'window_length' : 15}}},
  
        'deltax_order_1' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 1}},
        'deltax_order_2' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 2}},
        'deltax_order_3' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 3}},
        
        'deltax_order_2_smoothed_05' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 5}}},
        'deltax_order_2_smoothed_11' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 11}}},
        'deltax_order_2_smoothed_15' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 15}}},
        'deltax_order_2_smoothed_21' : {'function' : apply_deltax_method, 'kwargs' : {'order' : 2, 'smoothing' : {'window_length' : 21}}},
        }
    
    # we can also build the array of data transformations dynamically
    techniques = ['deltax', 'F_x']
    orders = [1, 2, 3, 4, 5]
    window_lengths = [0, 5, 11, 15, 21, 25]
    
    data_transformations = {}
    for t in techniques :
        for o in orders :
            for w in window_lengths :
                key = t + "_order_" + str(o)
                if w != 0 :
                    key += "_smoothed_w" + str(w)
                    
                arguments = {}
                arguments['function'] = apply_deltax_method
                if t == 'F_x' :
                    arguments['function'] = apply_euler_method_smoothing
                    
                arguments['kwargs'] = {}
                arguments['kwargs']['order'] = o
                if w != 0 :
                    arguments['kwargs']['smoothing'] = {'window_length' : w}
                    
                data_transformations[key] = arguments
    
    state_variables = [k for k in dictionary_trajectory if k != 't']
    t = dictionary_trajectory['t']
    
    for dt in data_transformations :
        
        if dt.startswith('F_x') :
            df = pd.DataFrame.from_dict(dictionary_trajectory)
            function = data_transformations[dt]['function']
            kwargs = data_transformations[dt]['kwargs']
            df_euler = function(df, **kwargs)
            
            # get the ground truth equations, compute values
            for state_variable, equation in equations.items() :
                delta_t = sympy.Symbol("Delta_t")
                euler_equation = sympy.Mul(delta_t, equation, evaluate=False)
                
                # prepare symbols and values for the lambdified equation
                equation_symbols = []
                for c in df_euler.columns :
                    if c == "delta_t" :
                        equation_symbols.append(sympy.sympify("Delta_t"))
                    else :
                        equation_symbols.append(sympy.sympify(c))
                symbol_values = [df_euler[c].values for c in df_euler.columns]
                
                # obtain predicted values
                equation_values = lambdify(equation_symbols, 
                                           euler_equation)(*symbol_values)
                
                # store the predicted values as additional columns
                df_euler["F_" + state_variable + "_pred"] = equation_values
            
            df_euler.to_csv(trajectory_name + "_" + dt + ".csv", index=False)
            
        elif dt.startswith('deltax') :
            df = pd.DataFrame.from_dict(dictionary_trajectory)
            function = data_transformations[dt]['function']
            kwargs = data_transformations[dt]['kwargs']
            df_deltax = function(df, **kwargs)
            
            # get the ground truth equations, compute values
            for state_variable, equation in equations.items() :
                equation_symbols = [c for c in df_deltax.columns 
                                    if not c.endswith("_deltax")]
                symbol_values = [df_deltax[c].values for c in equation_symbols]
                equation_values = lambdify(equation_symbols, equation)(*symbol_values)
                
                df_deltax[state_variable + "_deltax_pred"] = equation_values
                
            df_deltax.to_csv(trajectory_name + "_" + dt + ".csv", index=False)
            
    return

def main() :
    # some hard-coded values
    # source file, in JSON
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    # results folder
    results_directory = "../local_results/" + os.path.basename(__file__)[:-3]
    # noise levels (check ODEFormer paper)
    noise_levels = [0.0, 0.01, 0.05]
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
                trajectory_name = os.path.join(system_directory, 
                                                    "trajectory-%d-noise-%.2f" % (trajectory_index, noise_level))
                df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
                df_trajectory.to_csv(trajectory_name + ".csv", index=False)
                
                # also plot the trajectory
                plot_trajectory(os.path.basename(system_directory),
                                dictionary_trajectory, 
                                trajectory_name + ".png")
                
                # and now, perform data transformation and compare against
                # the ground truth
                compare_data_transformations(equations,
                                             dictionary_trajectory, 
                                             trajectory_name)
                
    # close log
    close_logging(logger)
    logging.shutdown()

if __name__ == "__main__" :
    sys.exit( main() )
