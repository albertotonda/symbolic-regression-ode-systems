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
import sys
import warnings # to silence annoying warnings

from sklearn.metrics import mean_squared_error, r2_score
from sympy import Symbol, sympify
from sympy.utilities.lambdify import lambdify
from typing import Dict, List, Union

# FFX should use the local version, all others have issues, closest to working properly is https://github.com/natekupp/ffx/tree/jmmcd-patch-1
from ffx.api import FFXRegressor
# ProGED can be installed from repository, pip install git+https://github.com/brencej/ProGED
from ProGED import EDTask, EqDisco
from ProGED.parameter_estimation import Estimator
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

# cut and pasted from the ODEFormer wrappers, with some minor modifications
def format_equation_syndy(expr: str, feature_names: List[str]):
    expr = regex.sub(r"(\d+\.?\d*) (1)", repl=r"\1 * \2", string=expr)
    for var_name in feature_names :
        expr = regex.sub(fr"(\d+\.?\d*) ({var_name})", repl=r"\1 * \2", string=expr)
    expr = expr.replace("^", "**")
    return expr

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
    
    # TODO it would be interesting to have a DataFrame with all the equations
    # obtained by each method, for each combination of hyperparameters; and
    # also using the two different systems of obtaining regular data,
    # the delta_y / delta_t and the dF/dDelta_t; the issue being, that every
    # approach has a different number and type of hyperparameters
    
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
        # where the 'X' are the trajectories, and the 'y' is the approximation
        # of the derivative for that variable. Other ways of manipulating the data
        # are also possible.
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
            
            # let's define the name of the output files
            state_variable_ffx_file = os.path.join(system_folder, "variable-%s-ffx.csv" % state_variable)
            
            # call the methods!
            
            # this is for FFX: an interesting note, if FFXRegressor is called
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
                    print("FFX failed on state variable \"%s\"!" % state_variable)
                    df_models = pd.DataFrame.from_dict({'equation' : ['Failure'], 
                                                        'error_message' : [str(ex)]})
                
                # in any case, save the DataFrame to file
                df_models.to_csv(state_variable_ffx_file, index=False)
            
            else :
                print("FFX file \"%s\" found, skipping to the next step..." % state_variable_ffx_file)
                             
                
        # now, let's try SINDy! This method does not require splitting up the
        # state variables one by one, or even the trajectories, but only formatting
        # the data to the appropriate format
        sindy_file = os.path.join(system_folder, "sindy.csv")
        
        if not os.path.exists(sindy_file) :
            print("Now running (py)SINDy...")
            
            # multiple trajectories are managed as a list (!) of ndarrays;
            # they also need to be transposed, as SINDy expects state_variables columns
            # and a number of rows equal to the size of the 't_train' time
            X_train_multi = []
            t_train_multi = []
            for trajectory in trajectories :
                X_train_multi.append(np.array(trajectory["y"]).T) # transposed
                t_train_multi.append(np.array(trajectory["t"]))
            
            model = SINDy()
            model.fit(X_train_multi, t=t_train_multi, multiple_trajectories=True)
            
            # create a DataFrame and save to disk
            df_sindy = pd.DataFrame.from_dict({'state_variable' : state_variables,
                                               'equation' : [format_equation_syndy(e, state_variables) for e in model.equations(precision=4)],
                                               'equation_syndy' : model.equations(precision=4)})
            df_sindy.to_csv(sindy_file, index=False)
        else :
            print("SINDy file \"%s\" found, skipping to the next step..." % sindy_file)
            
        # and finally, here is ProGED. ProGED can perform regression on either
        # single state variables one by one, or on the whole system at once;
        # in the ODEFormer publication, they used the second possibility, so
        # let's proceed in the same way. There is however an important caveat:
        # ProGED cannot work on two trajectories at the same time, it has to
        # deal with them one by one (!), then the models created for each
        # trajectory are merged
        proged_file = os.path.join(system_folder, "proged.csv")
        
        if not os.path.exists(proged_file) :
            print("Now running ProGED...")
            
            # we first need to create a pandas DataFrame with all variables and
            # time, shaped in the appropriate way; in particular,
            # we need a column named 't' with the time. So, first let's create
            # a numpy matrix with all values (times, variable values)
            for trajectory in trajectories :
                times = np.array(trajectory["t"])
                state_variable_values = np.array(trajectory["y"]).T # transposed
 
                # hstack puts the two np arrays side by side
                df_proged = pd.DataFrame(
                    np.hstack((times.reshape(-1,1), state_variable_values)), 
                    columns=['t'] + state_variables,
                )
                print(df_proged)
            
                # then, initialize an instance of the Equation Discoverer;
                # before that, however, we need to instantiate an instance of
                # EDTask, that describes the task more in detail
                #task = EDTask(data=df_proged, 
                #              lhs_vars=state_variables,
                #              lhs_vars=state_variables,
                #              rhs_vars=state_variables,
                #              task_type="differential",
                #              )
                
                ed = EqDisco(#task=task, # this does not seem to work
                             data=df_proged,
                             lhs_vars=state_variables,
                             rhs_vars=["t"] + state_variables,
                             system_size=len(state_variables),
                             task_type="differential",
                             sample_size=16, # same value as in ODEFormer baseline comparison
                             generator_template_name="polynomial", # same value as in ODEFormer
                             strategy_settings={'max_repeat' : 100},
                             verbosity=1,
                             )
                
                # once the instance is set up, there are two parts: model generation and model fitting
                proged_models = []
                try :
                    ed.generate_models()
                    proged_models = ed.models
                    print("Model generation successfully completed.")
                    
                except Exception as e :
                    print("Error occurred: \"%s\"" % str(e))
                
                # check if model generation has been successfully completed
                if len(proged_models) > 0 :
                    
                    # now, the issue here is that model parameter fitting can
                    # fail spectacularly, and crash the whole thing; but not
                    # for all models, so we first create an instance of an
                    # object that fits ProGED model parameters
                    estimator = Estimator(data=df_proged, settings=ed.estimation_settings)
                    
                    # and then, we fit the models one by one, just giving up
                    # on the models that do not work
                    proged_fitted_models = []
                    for model in proged_models :
                        try :
                            with warnings.catch_warnings() :
                                warnings.simplefilter("ignore") # ignore warnings
                                
                                print("Trying to fit model:", model)
                                fitted_model = estimator.fit_one(model)
                                proged_fitted_models.append(fitted_model)
                                print("Model successfully fitted.")
                        except Exception as e :
                            print("Error occurred: \"%s\", skipping to the next ProGED model..." % str(e))
                            proged_fitted_models.append(["" for i in range(0, len(state_variable))])
                    
                    print("ProGED fitted models:", proged_fitted_models)
                    
                    # now, at the end of this step, we should build a DataFrame
                    # with all the results;
                    dictionary_results = {sv : [] for sv in state_variables}
                    for sv in state_variables :
                        dictionary_results[sv + "_parametrized"] = []
                        
                    for i in range(0, len(proged_models)) :
                        for j in range(0, len(state_variables)) :
                            sv = state_variables[j]
                            dictionary_results[sv].append(str(proged_models[i].expr[j]))
                            
                            # now, here a fitted model can be an instance of Model containing
                            # several expressions, or just a list of strings; so
                            # we first need to check that
                            model_fitted = proged_fitted_models[i]
                            #print("Model #%d, fitted: %s" % (i, str(model_fitted)))
                            
                            model_fitted_sv = None
                            if not isinstance(model_fitted, list) :
                                model_fitted_sv = str(model_fitted.full_expr()[j])
                            else :
                                model_fitted_sv = str(model_fitted[j])
                            
                            dictionary_results[sv + "_parametrized"].append(model_fitted_sv)
                    
                    print("Saving results to file \"%s\"..." % proged_file)
                    df_proged_results = pd.DataFrame.from_dict(dictionary_results)
                    df_proged_results.to_csv(proged_file, index=False)
                    
                sys.exit(0) # TODO remove this
            
            #sys.exit(0) # TODO remove this
            
            
            
        