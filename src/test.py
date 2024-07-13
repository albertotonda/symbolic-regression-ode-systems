# This is a global test for the whole pipeline:
# - read an ODE system (plus initial conditions, delta_t, etc.) from file
# - obtain dataframe with the values
# - apply explicit Euler method, get modified data set as dataframe
# - learn equations one by one
# - put equations together in a system
# - optimize parameters

import datetime
import juliacall

from pysr import PySRRegressor
from scipy import integrate
from sympy.parsing.sympy_parser import parse_expr

# local libraries
from create_dataset_from_ode_system import get_df_from_ode_system, parse_ode_from_text
from explicit_euler_method import apply_euler_method
from learn_equations import prune_equations
from optimize_ode_systems import dX_dt
from local_utility import MyTimeoutError, run_process_with_timeout


if __name__ == "__main__" :
    
    # some local imports
    import itertools
    import multiprocessing
    import os
    import pandas as pd
    import pickle
    import shutil
    import sys
    
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from sklearn.metrics import mean_squared_error, r2_score
    
    # this is just a test with Lotka-Volterra (or another ODE system); 
    # here are some hard-coded values
    random_seed = 42
    timeout = 120 # in seconds
    #ode_system_file_name = "../data/lotka-volterra.txt"
    ode_system_file_name = "../data/rossler-stable.txt"
    # percentage of generated data to use for each split
    training_split = 0.5
    validation_split = 0.25
    test_split = 0.25
    
    print("Starting experiment with ODE system in \"%s\"..." % ode_system_file_name)
    # 1. create an experiment folder; if a folder containing the same name is already
    # inside the root folder, skip
    ode_system_name = os.path.basename(ode_system_file_name).split(".")[0]
    sub_directories = [ x[0] for x in os.walk(".") if x[0].find(ode_system_name) != -1]
    
    results_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + ode_system_name
    
    if len(sub_directories) > 0 :
        results_folder = sub_directories[0]
        print("Found folder \"%s\", resuming from files inside..." % results_folder)
    else :
        print("Creating new folder for the experiment, \"%s\"..." % results_folder)
        os.mkdir(results_folder)
        
    # copy the original file inside the folder, if it is not there yet
    destination_ode_system_file_name = os.path.join(results_folder, os.path.basename(ode_system_file_name))
    if not os.path.exists(destination_ode_system_file_name) :
        shutil.copyfile(ode_system_file_name, destination_ode_system_file_name)
    
    # 2. check if inside the experiment folder there is already a CSV file with
    # the original data from the ODE system; if so, skip
    # this below is the file name that we are expecting to find
    solved_ode_file_name = os.path.join(results_folder, ode_system_name + ".csv")
    df_ode = None
    if not os.path.exists(solved_ode_file_name) :    
        print("Obtaining data for ODE system in file \"%s\"..." % ode_system_file_name)
        equations, ordered_variables, initial_conditions, time_step, max_time = parse_ode_from_text(ode_system_file_name)
        df_ode = get_df_from_ode_system(equations, ordered_variables, initial_conditions, time_step, max_time)
        df_ode.to_csv(solved_ode_file_name, index=False)
    else :
        print("File with data from ODE system found, loading it and skipping to the next step...")
        df_ode = pd.read_csv(solved_ode_file_name)
        
    # split data between train, validation, and test; we are going consecutively
    # as this is a time series
    data_set_size = df_ode.shape[0]
    df_ode_train = df_ode.iloc[:int(training_split * data_set_size)]
    df_ode_val = df_ode.iloc[int(training_split * data_set_size):int(training_split * data_set_size) + int(validation_split * data_set_size)]
    df_ode_test = df_ode.iloc[int(training_split * data_set_size) + int(validation_split * data_set_size):]
    # this last data set includes training + validation
    df_ode_train_val = df_ode.iloc[:int(training_split * data_set_size) + int(validation_split * data_set_size)]
    
    print("Total data set size:", df_ode.shape[0])
    print("Training samples:", df_ode_train.shape[0])
    print("Validation samples:", df_ode_val.shape[0])
    print("Test samples:", df_ode_test.shape[0])
        
    # 3. check if inside the experiment folder there is already a CSV file with
    # the data modified with the explicit Euler method; if so, skip
    euler_method_file_name = os.path.join(results_folder, ode_system_name + "-euler.csv")
    
    if not os.path.exists(euler_method_file_name) :
        print("Applying explicit Euler method...")
        df_euler = apply_euler_method(df_ode_train)
        df_euler.to_csv(euler_method_file_name, index=False)
    else :
        print("File with result of explicit Euler method found, loading it and skipping to the next step...")
        df_euler = pd.read_csv(euler_method_file_name)
    
    # 4. for each new variable to perform regression on, check if the corresponding
    # CSV file with the equations exists in the folder; if so, skip to the next variable
    target_names = [c for c in df_euler.columns if c.startswith("F_")]
    dictionary_regressors = {} # used to store the full regressor objects
    dictionary_equations = {}
    
    for target in target_names :
        target_equations_file_name = os.path.join(results_folder, "equations-%s.csv" % target)
        target_regressor_file_name = os.path.join(results_folder, "regressor-%s.pkl" % target)
        
        if not os.path.exists(target_equations_file_name) or not os.path.exists(target_regressor_file_name) :
            print("Now running symbolic regression for variable \"%s\"..." % target)
            
            # create dataframes with selection: TODO split data? cross-validation?
            y = df_euler[target]
            X = df_euler[[c for c in df_euler.columns if c != target and not c.startswith("F_")]]
            
            # initialize PySRRegressor
            symbolic_regressor = PySRRegressor(
                population_size=50, # for the first experiment, it was 50
                niterations=1000,
                batching=True, # use batches instead of the whole dataset
                batch_size=50, # 50 is the default value for the batches
                model_selection="best",  # Result is mix of simplicity+accuracy
                binary_operators=["+", "*", "/", "-", ],
                unary_operators=["sin", "cos", "exp", "log", "sqrt",],
                early_stop_condition=("stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"), # stop early if we find a good and simple equation
                temp_equation_file=True, # does not clutter directory with temporary files
                verbosity=1,
                random_state=random_seed,
                )
            
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
            
    # 5. at this point, we need to "prune" the equations; we first derivate each
    # in "delta_t", then set "delta_t=0" and remove the copies
    dictionary_pruned_equations = {}
    for target in target_names :
        variable_name = target[2:] # the beginning of 'target' is 'F_', so 'F_x' -> 'x'
        pruned_equations_file_name = os.path.join(results_folder, "equations-pruned-%s.csv" % variable_name)
        
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
    
    # 6. next step: the pruned equations are paired to recreate candidate ODE systems
    # as text files, that will be later read to optimize their parameters
    equations_list = [ equations for variable, equations in dictionary_pruned_equations.items() ]
    variables_list = [ variable for variable, equations in dictionary_pruned_equations.items() ]
    print(equations_list)
    candidate_ode_systems = list(itertools.product(*equations_list))
    print("Created a total of %d candidate ODE systems..." % len(candidate_ode_systems))
    
    candidate_systems_folder = os.path.join(results_folder, "candidate_systems/")
    if not os.path.exists(candidate_systems_folder) :
        os.mkdir(candidate_systems_folder)
    
    for index, system in enumerate(candidate_ode_systems) :
        system_file_name = os.path.join(candidate_systems_folder, "candidate-ode-system-%d.txt" % index)
        if not os.path.exists(system_file_name) :
            system_text = ""
            for i in range(0, len(system)) :
                system_text += "d" + str(variables_list[i]) + "/dt" + " = "
                system_text += str(system[i]) + "\n"
            
            with open(system_file_name, "w") as fp :
                fp.write(system_text)
                
    # 6b. however, we could also check what is the default equation picked by PySRRegressor
    # for each state variable, building a candidate ODE system with just the best
    candidate_system_default_equations = []
    candidate_system_default_equations_text = ""
    for target in target_names :
        eq = prune_equations([dictionary_regressors[target].sympy()])[0]
        candidate_system_default_equations.append(eq)
        
        candidate_system_default_equations_text += "d" + target[2:] + "/dt" + " = "
        candidate_system_default_equations_text += str(eq) + "\n"
    
    with open(os.path.join(results_folder, "candidate_default_equations.txt"), "w") as fp :
        fp.write(candidate_system_default_equations_text)
    
    # 7. now, run the systems against the original
    # data, checking what is their original performance, and which one
    # would be the best without parameter optimization
    candidates_initial_performance_file_name = os.path.join(results_folder, "candidates_initial_performance.csv")
    
    # TODO for long experiments (thousands of candidate systems) it is better
    # to save the intermediate results of each candidate in a single file;
    # so, if the file exists, we should read it; check what is the id of the
    # last candidate system evaluated; and restart from there
    # also, we need to save the file after each evaluation
    
    # whatever we did so far, inside 'candidate_ode_systems' we now got a list
    # of n-uples of equations, each one representing a variable in the ODE system;
    # so, we prepare a dictionary in any case, if the file with the results exists,
    # we read it and replace the dictionary 
    dictionary_unoptimized_candidate_performances = {'system_id' : [], 'MSE' : [], 'R2' : []}
    
    # also prepare all the other necessary variables
    # and we can go back and read the original values from df_ode; but
    # now, we are going to consider the training samples + the validation samples
    initial_conditions = [df_ode_train_val[v].values[0] for v in variables_list]
    t = df_ode_train_val["t"].values
    measured_values = df_ode_train_val[variables_list].values
    current_candidate_index = 0
    
    if os.path.exists(candidates_initial_performance_file_name) :
        print("File \"%s\" exists, reading and resuming candidate system evaluation..." %
              candidates_initial_performance_file_name)
        # if the file exists, read it and transform it to a dictionary
        df_initial_performance = pd.read_csv(candidates_initial_performance_file_name)
        dictionary_unoptimized_candidate_performances = df_initial_performance.to_dict(orient='list')
        #print(dictionary_unoptimized_candidate_performances)
        
        # find the highest index in the column 'system_id'; we should evaluate
        # the following system
        current_candidate_index = df_initial_performance["system_id"].max() + 1
        print("Evaluation will be resumed from candidate %d/%d" % 
              (current_candidate_index, len(candidate_ode_systems)))
        
    # now, let's iterate over the candidate ODE systems; if the index was read from
    # an existing file, it could be in principle high enought to directly skip
    # the following block of code
    while current_candidate_index < len(candidate_ode_systems) :
        
        candidate_ode_system = candidate_ode_systems[current_candidate_index]
        
        # the dX_dt function needs a dictionary of variable -> symbolic expression
        candidate_equations = {variables_list[i] : parse_expr(candidate_ode_system[i])
                               for i in range(0, len(variables_list))}
                  
        # prepare arguments for function scipy.integrate.odeint
        args = (dX_dt, initial_conditions, t)
        kwargs = {'args' : (candidate_equations, variables_list, ),
                  'full_output' : True}
        
        # SIXTH ATTEMPT at managing timeouts; let's go back to multiprocessing
        result = None
        try :
            result = run_process_with_timeout(integrate.odeint, timeout, args, kwargs)
        except TimeoutError as e :
            print(e)
        except Exception as e :
            print("Exception:", str(e))
        
        # these are the default values that will be used in case of errors
        candidate_mse = sys.float_info.max
        candidate_r2 = -sys.float_info.max
        
        # if the integration succeeded, the result is not None; however,
        # there might still be the case that some issues appeared, for example
        # Y might contain infinite/NaN values. A control is thus needed.
        if result is not None :
            try :
                Y, info_dict = result
                candidate_mse = mean_squared_error(measured_values, Y)
                candidate_r2 = r2_score(measured_values, Y)
            except Exception as e :
                print("Exception raised while recovering integration result:", str(e))
                 
        # add information to the dictionary
        dictionary_unoptimized_candidate_performances["system_id"].append(current_candidate_index)
        dictionary_unoptimized_candidate_performances["MSE"].append(candidate_mse)
        dictionary_unoptimized_candidate_performances["R2"].append(candidate_r2)
        
        print("Candidate %d, R2=%.4f" % (current_candidate_index, candidate_r2))
        
        # save the results to a file after every evaluation
        df_initial_performance = pd.DataFrame.from_dict(dictionary_unoptimized_candidate_performances)
        df_initial_performance.to_csv(candidates_initial_performance_file_name, index=False)
        
        # increase index
        current_candidate_index += 1
    
    # a final save, sorting the candidate systems by performance
    df_initial_performance = pd.DataFrame.from_dict(dictionary_unoptimized_candidate_performances)
    df_initial_performance.sort_values(by="R2", inplace=True)
    df_initial_performance.to_csv(candidates_initial_performance_file_name, index=False)