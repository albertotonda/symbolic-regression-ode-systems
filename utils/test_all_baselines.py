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

if __name__ == "__main__" :
    
    # hard-coded values
    odebench_json_file_name = "../data/odebench/all_odebench_trajectories.json"
    random_seed = 42
    selected_system_ids = [24] # if this is empty, it just gets all systems
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
        
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        print("The system has state variables:", state_variables)
        
        # so, here we get the two trajectories, then compute the derivatives
        # for both trajectories, and eventually use them for training
        trajectories = system["solutions"][0]
        
        # for most methods, we have to perform a variable-by-variable regression,
        # where the 'X' are the trajectories, and the 'y' is the approxiamation
        # of the derivative for that variable
        for state_variable_index, state_variable in enumerate(state_variables) :
            
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
                
            # now the data for the state variable has been collected
            print("X.shape=", X.shape)
            print("y.shape=", y.shape)
            
            # call the methods! an interesting note, if FFXRegressor is called
            # with a Pandas DataFrame as X, it will also automatically obtain the
            # names of the features from the columns; so, let's convert X to a DataFrame
            df_X = pd.DataFrame.from_dict({state_variables[i] : X[:,i] for i in range(0, len(state_variables))})
            ffx_regressor = FFXRegressor()
            ffx_regressor.fit(df_X, y)
            print(ffx_regressor.models_)
            
        
        