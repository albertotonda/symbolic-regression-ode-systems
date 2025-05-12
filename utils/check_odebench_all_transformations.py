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
from dcode import basis, gppca

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

def apply_dcode_method(df_trajectory, r=-1, sigma_in=0.1, 
                       sigma_in_mul=None, freq_int=20, 
                       n_basis=50, basis=basis.FourierBasis, 
                       smoothing=None) :
    """
    This function attempts to apply the methodology from the D-CODE paper.
    """
    # this is quite complex; inside the D-CODE files, the flow is as follows:
    # - run_simulation_vi contains the whole structure
    # - starting from the trajectory data, which in D-CODE is generated by
    #   a proprietary function
    # - interpolate.get_ode_data() will process the trajectory and return the 
    #   modified data set
    # - however, the output is NOT inside X_ph, y_ph (which are all zero), it's
    #   inside the ode_data dictionary
    # - all the parts inside the ode_data dictionary are then combined in the
    #   fitness function, which is inside _program.Program.raw_fitness()
    # - that function actually returns TWO values, one is the fitness, the other
    #   is the OOB fitness; but the two, called from genetic.py , _parallel_evolve(),
    #   seem to actually be the same (!!!)
    # - now, the last issue is that there are actually A TON of hyperparameters!
    #   and it looks like that each ODE has its own hyperparameters (...)
    # - hyperparameters I could find
    #       -- noise_sigma=[0.09, 0.15, 0.2, 0.25, 0.3], used by Gaussian Process interpolation; 
    #       -- r=[-1, 2, 3, 4, 7], part of ode_config
    #       -- sigma_in=[0.1, 0.15, 0.2]
    #       -- sigma_in_mul=[2.0] ; but it sometimes replaces sigma_in_mul :-D
    #       -- freq_int=[20, 100, 730]
    #       -- n_basis=[50] ; weird, the paper claimed 60-70 (!)
    #       -- basis=[basis.FourierBasis, basis.CubicSplineBasis]
    
    # it would probably be better to fully rewrite the whole thing; so here is
    # an attempt! let's first get the time and the corresponding trajectory values
    state_variables = [c for c in df_trajectory.columns if c != 't']
    t = df_trajectory['t'].values
    yt = df_trajectory[state_variables].values
    
    # let's create a dictionary
    dictionary_dcode = {}
    dictionary_dcode['t'] = t
    for c in state_variables :
        dictionary_dcode[c] = df_trajectory[c].values
    
    # now, we can create all parts required by the fitness function; this code
    # below might look a bit inefficient, but it's lifted directly from the D-CODE repo
    for v in range(0, yt.shape[1]) :
        state_variable = state_variables[v]
        y_v = yt[:,v]
        T = t[-1] # max value of time appearing in the trajectory
        
        # first mysterious hyperparameter
        if r < 0 :
            r = 1 # used to be; shape [1] of the 3d vactor, with dimension 
                  # containing results for different initial conditions
        
        # sigma! what is it? why is it obtained dividing by the integrator's frequence?
        # I have no idea, and maybe this should just take into account given 'sigma'
        #if sigma_in_mul is not None :
        #    sigma_in = sigma_in_mul / freq
        # otherwise, we keep the regular sigma_in
        
        # now, we need to test; if there is no noise, we can just use the noise-free
        # version, which is much easier; actually, we could employ a variant and
        # just use pysindy with the settings to generate a smoothed trajectory
        
        
        if smoothing is not None :
            print("Smoothing the data using the SG filter...")
            smoothing["polyorder"] = 3
            smoothing["axis"] = 0
            y_v = savgol_filter(y_v, **smoothing)
            
        # first, create some weights, giving more importance to beginning and end
        weight = np.ones_like(t)
        weight[0] /= 2
        weight[-1] /= 2
        weight = weight / weight.sum() * T
        
        # let's go with computing all the ingredients of the mystical transformation
        # again, the code might look weird, but it's just copy and pasted
        basis_func = basis(T, n_basis)
        
        # here are all the 'ingredients': g_dot, g, c
        g_dot = basis_func.design_matrix(t, derivative=True)
        g = basis_func.design_matrix(t, derivative=False)
        c = (y_v * weight[:, None]).T @ g_dot
        x_hat = y_v # the 
        
        #print("Shapes of the ingredients:")
        #print("x_hat=", x_hat.shape)
        #print("c=", c.shape)
        #print("g=", g.shape)
        #print("g_dot=", g_dot.shape)
        
        # let's add pieces to the dictionary!
        dictionary_dcode[state_variable + '_x_hat'] = x_hat
        
        # also add the integration weights
        dictionary_dcode[state_variable + "_integration_weights"] = weight
        
        # the other 'ingredients' actually have multiple parts
        for g_index in range(0, g.shape[1]) :
            dictionary_dcode[state_variable + '_g_%d' % g_index] = g[:,g_index]
            
        for g_dot_index in range(0, g_dot.shape[1]) :
            dictionary_dcode[state_variable + "_g_dot_%d" % g_dot_index] = g_dot[:,g_dot_index]
        
        for c_index in range(0, c.shape[1]) :
            dictionary_dcode[state_variable + "_c_%d" % c_index] = c[:,c_index]
            
    df_dcode = pd.DataFrame.from_dict(dictionary_dcode)
    
    return df_dcode

def compute_dcode_fitness_function(equation, symbols, x_hat, integration_weights, c, g) :
    """
    This function implements the computation of the DCODE fitness value for a
    target equation.
    - equation is equation in symbolic form (sympy)
    - x_hat is the original trajectory with shape (time_instants x features);
      IMPORTANT NOTE: features need to appear in the same order as the symbols
      in the symbolic equation, normally the original order of columns in the
      CSV file used for the experiments (x_0, x_1, ..., x_N)
    - symbols is a list of sympy symbols contains all symbols in the equations
    - c are the precomputed values of the DCODE term C, shape (time_instants x n_support_functions)
    - g are the precomputed values of the DCODE term g, shape (time_instants x n_support_functions)
    """
    #print("Shape of x_hat:", x_hat.shape)
    #print("Shape of integration_weights:", integration_weights.shape)
    #print("Shape of c:", c.shape)
    #print("Shape of g:", g.shape)
    # get the values of the symbols in the equation (in the base case scenario,
    # this should be completely equivalent to x_hat, with maybe a different shape,
    # but just in case, let's recompute it here)
    symbol_values = [x_hat[:,i] for i in range(0, x_hat.shape[1])]
    
    # obtain the value of the function
    y_hat = lambdify(symbols, equation)(*symbol_values)
    #print("Shape of y_hat:", y_hat.shape)
    partial = (y_hat * integration_weights[:, None]).T
    #print("Shape of (y_hat * integration_weights).T", partial.shape)
    
    # let's compute the c_hat values for the current equation
    c_hat = (y_hat * integration_weights[:, None]).T @ g
    print("c_hat:", c_hat)
    
    #print("Shape of c_hat:", c_hat.shape)
    #print("Shape of c:", c.shape)
    
    # fitness value (the original formula also had weights, but if I am not 
    # mistaken the original code set them all at [1.0, ..., 1.0])
    sample_weight = np.ones(y_hat.shape)
    fitness_value = np.sum((c + c_hat) ** 2 * sample_weight[:, None]) / np.sum(sample_weight)
    #fitness_value = np.sum((c + c_hat) ** 2)
    
    return fitness_value
    
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
    
    # TODO remove this, it's a horrible temporary hack
    if False :
        data_transformations = {
            'C_x' : {'function' : apply_dcode_method, 
                     'kwargs' : {'n_basis' : 50, }},
            'C_x_smoothed_w15' : {'function' : apply_dcode_method,
                                  'kwargs' : {'n_basis' : 50,
                                  'smoothing' : {'window_length' : 15}}}, 
        }
        
    data_transformations['C_x'] = {'function' : apply_dcode_method, 
                                   'kwargs' : {'n_basis' : 50, }}
    data_transformations['C_x_smoothed_w15'] = {'function' : apply_dcode_method,
                          'kwargs' : {'n_basis' : 50,
                          'smoothing' : {'window_length' : 15}}}
    
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
            
        elif dt.startswith('C_x') :
            df = pd.DataFrame.from_dict(dictionary_trajectory)
            function = data_transformations[dt]['function']
            kwargs = data_transformations[dt]['kwargs']
            df_dcode = function(df, **kwargs)
            
            for state_variable, equation in equations.items() :
                equation_symbols = [v for v in equations] # all state variables
                
                c = df_dcode[[c for c in df_dcode.columns 
                             if c.startswith(state_variable + "_c_")]].values
                g = df_dcode[[c for c in df_dcode.columns 
                             if c.startswith(state_variable + "_g_") and
                             c.find("g_dot") == -1]].values
                x_hat = df_dcode[[c for c in df_dcode.columns
                                  if c.endswith("_x_hat")]].values
                integration_weights = df_dcode[state_variable + "_integration_weights"].values
                
                dcode_fitness_value = compute_dcode_fitness_function(equation, 
                                                                     equation_symbols, 
                                                                     x_hat, 
                                                                     integration_weights, 
                                                                     c, 
                                                                     g)
                
                # now, differently from deltax and Fx, Cx does not compute a
                # target value for each entry in the table, but just one general
                # value; so, let's just have a column with the C_x value repeated
                df_dcode["C_x_" + state_variable] = [dcode_fitness_value] * df_dcode.shape[0]
                        
            df_dcode.to_csv(trajectory_name + "_" + dt + ".csv", index=False)
            
            
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
