"""
This is a script to test all other baseline algorithms on a JSON file, in the
output format used by ODEBench.

2024-07-29
----------
The ODEFormer repository already had some code to perform this very same test,
but unfortunately it is not working, probably due to an issue between two branches,
plus some other problems of compatibility between Linux and Windows. Trying to
just use the wrappers alone creates further issues with package numbalsoda, so
the only alternative I found is to just rewrite the whole thing.

Still, I could use the ODEFormer code as a starting point.
"""
import json
import numpy as np
import os
import pandas as pd
import re as regex

from sklearn.metrics import mean_squared_error, r2_score
from sympy import Symbol, sympify
from sympy.utilities.lambdify import lambdify
from typing import Dict, List, Union

from ffx import FFXRegressor # install this using the https://github.com/natekupp/ffx/tree/jmmcd-patch-1 branch
from pysindy import ConcatLibrary, CustomLibrary, PolynomialLibrary, SINDy, optimizers
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference

# these two functions below are a cut/paste from ODEFormer
def approximate_derivative(
    trajectory: np.ndarray,
    times: np.ndarray, 
    finite_difference_order: Union[None, int] = 2,
    smoother_window_length: Union[None, int] = None,
) -> np.ndarray :
    times = times.squeeze()
    assert len(times.shape) == 1 or np.all(times.shape == trajectory.shape), f"{times.shape} vs {trajectory.shape}"
    
    fd = get_differentiation_method(
        finite_difference_order = finite_difference_order, 
        smoother_window_length = smoother_window_length,
    )
    return fd._differentiate(trajectory, times)

def get_differentiation_method(
    finite_difference_order: Union[None, int] = None, 
    smoother_window_length: Union[None, int] = None,
) :
    if finite_difference_order is None:
        finite_difference_order = 2
    if smoother_window_length is None:
        return FiniteDifference(order=finite_difference_order)
    return SmoothedFiniteDifference(
        order=finite_difference_order,
        smoother_kws={'window_length': smoother_window_length},
    )

# these below are functions that I wrote to make my life easier (hopefully)
def score_models_on_training(df_X, y, models, metrics=[mean_squared_error, r2_score]) :
    """
    Given the training data and the ground truth, score the models; returns a
    dictionary of lists, ready to be converted to a pandas DataFrame.
    """
    dictionary = {'equation': []}
    for metric in metrics :
        dictionary[metric.__name__] = []
    
    for model in models :
        # transform the equation in a symbolic Sympy expression
        eq = sympify(model)
        symbols = [Symbol(c) for c in df_X.columns]
        symbol_values = [df_X[c].values for c in df_X.columns]
        
        # and now, let's lambdify and run it! there are some corner cases where
        # this fails, probably due to the use of a 'max' operation; still, it is
        # possible to correct this, using a slower way of iterating over the
        # DataFrame, passing the arguments as tuples of floating points
        try :
            y_pred = lambdify(symbols, eq)(*symbol_values)
        except ValueError :
            eq_lambda = lambdify(symbols, eq)
            
            y_pred = np.zeros(y.shape)
            for index, row in df_X.iterrows() :
                symbol_values = [row[c] for c in df_X.columns]
                y_pred[i] = eq_lambda(*symbol_values)
        
        # now, if the equation is a constant, for some reason lambdify returns
        # just ONE value, no matter the size of the arguments; no worries, we
        # can easily check this and correct it
        if not isinstance(y_pred, np.ndarray) :
            y_pred = y_pred * np.ones(y.shape)
        
        # store all relevant information inside the dictionary
        dictionary["equation"].append(str(model))
        
        for metric in metrics :
            dictionary[metric.__name__].append(metric(y, y_pred))
        
    return dictionary

if __name__ == "__main__" :
    
    # hard-coded values
    odebench_json_file_name = "../data/odebench/all_odebench_trajectories.json"
    random_seed = 42
    selected_system_ids = [s for s in range(24, 64)] # if this is an empty list, it just gets all systems
    results_folder = "../local_results/results-odebench-baselines"
    
    # read the file containing all the trajectories
    print("Reading file \"%s\", containing the trajectories..." % odebench_json_file_name)
    odebench = None
    with open(odebench_json_file_name, "r") as fp :
        odebench = json.load(fp)
        
    # if no system id has been specified, we are going to work on all of them
    if len(selected_system_ids) == 0 :
        selected_system_ids = [system["id"] for system in odebench]
    
    odebench = [system for system in odebench if system["id"] in selected_system_ids]
    
    # create output directory
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    
    # start the loop
    for system in odebench :
        print("Now working on system %d (\"%s\")" % (system["id"], system["eq_description"]))
        
        # check if the system folder already exists, and if it does not, create it
        system_folder = os.path.join(results_folder, "system-%d" % system["id"])
        if not os.path.exists(system_folder) :
            os.makedirs(system_folder)
        
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        print("The system has state variables:", state_variables)
        
        # so, here we get the two trajectories, then compute the derivatives
        # for both trajectories, and eventually use them for training
        trajectories = system["solutions"][0]
        
        # for most methods, we have to perform a variable-by-variable regression,
        # where the 'X' are the trajectories, and the 'y' is the approxiamation
        # of the derivative for that variable
        for state_variable_index, state_variable in enumerate(state_variables) :
            print("Now working on state variable \"%s\"..." % state_variable)
            # prepare all the data in a single np.array
            X = None
            y = None
            for trajectory in trajectories :
                
                # get time and the current state variable values separately
                trajectory_time = np.array(trajectory["t"])
                trajectory_state_variable = np.array(trajectory["y"][state_variable_index])
                
                # get a numerical approximation of the derivative
                derivative = approximate_derivative(trajectory_state_variable, trajectory_time)
                
                # also, organize all state variables in an appropriately shaped array
                trajectory_state_variables = np.zeros((trajectory_state_variable.shape[0], len(state_variables)))
                for i in range(0, len(state_variables)) :
                    trajectory_state_variables[:,i] = np.array(trajectory["y"][i])
                
                # store the information
                if X is None :
                    X = trajectory_state_variables
                else :
                    X = np.concatenate((X, trajectory_state_variables), axis=0)
                    
                if y is None :
                    y = derivative
                else :
                    y = np.concatenate((y, derivative), axis=0)
            
            # let's define the name of the output file
            state_variable_ffx_file = os.path.join(system_folder, "variable-%s-ffx.csv" % state_variable)
            
            # call the methods! an interesting note, if FFXRegressor is called
            # with a Pandas DataFrame as X, it will also automatically obtain the
            # names of the features from the columns; so, let's convert X to a DataFrame
            if not os.path.exists(state_variable_ffx_file) :
                # we need to use a try/except scope here, in some cases it just
                # crashes with a SystemExit
                try :
                    df_X = pd.DataFrame.from_dict({state_variables[i] : X[:,i] for i in range(0, len(state_variables))})
                    ffx_regressor = FFXRegressor()
                    ffx_regressor.fit(df_X, y)
                    model_strings = [str(m) for m in ffx_regressor.models_]
                    print(model_strings)
                    
                    # score the models, obtain a dictionary, save the dictionary to file
                    dictionary_models = score_models_on_training(df_X, y, model_strings)
                    df_models = pd.DataFrame.from_dict(dictionary_models)
                
                except SystemExit as ex :
                    df_models = pd.DataFrame.from_dict({'equation' : ['Failure'], 
                                                        'error_message' : [str(ex)]})
                
                # in any case, save to file
                df_models.to_csv(state_variable_ffx_file, index=False) 
        
        