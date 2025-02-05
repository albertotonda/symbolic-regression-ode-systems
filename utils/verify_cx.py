# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:05:52 2025

This script is trying to validate the C_x approach, using their original data,
but also attempting to re-implement the evaluation in a more convenient external
python function.

@author: Alberto
"""
import json
import numpy as np
import os
import pickle
import re as regex
import sympy
import sys

from sympy.utilities.lambdify import lambdify

# brutally add the main directory of D-CODE to the python path
sys.path.append("../../D-CODE-ICLR-2022")
from basis import FourierBasis

def main_lorenz() :
    # NOTE: for the moment, this code works on the Lorenz system, ***ONLY***
    # hard-coded values
    target_folder = "../../D-CODE-ICLR-2022/results_vi/Lorenz/noise-0.0/sample-50/freq-25"
    target_file = "grad_seed_0.pkl"

    # load pickle
    with open(os.path.join(target_folder, target_file), "rb") as fp :
        pickle_dictionary = pickle.load(fp)
        print(pickle_dictionary)
        
    # parse it into its different components
    ode_data = pickle_dictionary['ode_data']
    print("ode_data:", ode_data)
    
    x_hat = ode_data['x_hat']
    print("Shape of x_hat:", x_hat.shape)
    
    t_new = pickle_dictionary['t_new']
    print("Shape of t_new:", t_new.shape)
    
    c = ode_data['c']
    print("Shape of c:", c.shape)
    
    g = ode_data['g']
    print("Shape of g:", g.shape)
    
    integration_weights = ode_data['weights']
    print("Shape of integration_weights:", integration_weights.shape)
    
    # here are the authentic expressions
    ground_truth_equations = pickle_dictionary['ode'].get_expression()
    print("Ground truth ODE system:", ground_truth_equations)
    # however, this returns the system WITHOUT the parameters/coefficients/constants...
    # so, let's write our own correct equation for x0
    ground_truth_x0 = "-10.0 * X0 + 10.0 * X1"
    # convert equation to sympy format (to later lambdify it)
    equation = sympy.sympify(ground_truth_x0)
    
    # TODO recompute fitness value of one of the gplearn expressions, and
    # ensure that it's coherent with the evaluation I added to gp_utils.run_gp_ode()
    # result of the experiment with seed 10
    # -10.0910291553439*X0 + 10.0910291553439*X1 -> fitness 11.420898998459684
    result_experiment_seed_10_x0 = "-10.0910291553439*X0 + 10.0910291553439*X1"
    equation = sympy.sympify(result_experiment_seed_10_x0)
    
    # now we should have an array with size (n_time_steps) x (n_initial_conditions) x (n_variables)
    # stored in x_hat; we follow the same steps used in the computation of the
    # fitness function, but we obtain the y_hat value using sympy instead of
    # just evaluating the SR expression inside a GP tree; this below is a cut/paste
    # of the code found in _program._Program.raw_fitness()
    T, B, D = x_hat.shape

    x_hat_long = x_hat.reshape((T * B, D))
    
    # this is the part which was modified
    #y_hat_long = self.execute(x_hat_long)
    symbols = [sympy.sympify("X%d" % i) for i in range(0, 3)]
    symbol_values = [x_hat_long[:,i] for i in range(0,3)]
    y_hat_long = lambdify(symbols, equation)(*symbol_values)
    # end of the part that was modified
    
    y_hat = y_hat_long.reshape((T, B))

    c_hat = (y_hat * integration_weights[:, None]).T @ g
    
    # sample_weight was an array of ones, even in the original
    sample_weight = np.ones(c.shape)
    
    c_x_fitness_value = np.sum((c + c_hat) ** 2 * sample_weight[:, None]) / np.sum(sample_weight)
    all_c_x_differences = c + c_hat
    
    print("C_x fitness for the Lorenz ground truth \"%s\": %.4f" %
          (str(equation), c_x_fitness_value))
    print("All c_x differences:", all_c_x_differences)
    print("ALl c_x differences, shape:", all_c_x_differences.shape)
    
    # now, this 'all_c_x_differences' represents the difference in values for
    # each C_x on each initial condition, for a shape (50, 50)
    for i in range(0, all_c_x_differences.shape[0]) :
        print("- For initial conditions #%d, sum of c_x squared differences: %.4f" %
              (i, np.sum(all_c_x_differences[i]**2)/all_c_x_differences.shape[1]))
     
    return
    
def main_odebench() :
    # now, in this other main, we are going to try the same approach, but on
    # an ODEBench trajectory; this, however, also requires computing the c_x
    # and all that jazz for the original trajectory
    
    # hard-coded values
    odebench_file = "../data/odebench/all_odebench_trajectories.json"
    n_basis = 50 # number of support functions
    basis = FourierBasis # type of support function
    
    # let's start by loading the odebench file
    print("Loading odebench trajectory file \"%s\"..." % odebench_file)
    odebench = json.load(open(odebench_file, "r"))
    
    for system in odebench :
        print("System %d: %s" % (system["id"], system["eq_description"]))
        
        # get equations
        # get the description of the state variables, and capture their names
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        # associate each variable with the expression of its derivative; for
        # some weird reason, the key "substituted" returns a list of ONE element (!)
        # that contains a list of strings ^_^;;
        equations = {state_variables[i] : sympy.sympify(system["substituted"][0][i])
                     for i in range(0, len(state_variables))}
        for var, eq in equations.items() :
            print("\t%s : %s" % (var, eq))
        
        # get trajectories, with an array shape which will be used later to compute c_x
        trajectories = system["solutions"][0]
        trajectories_array = None
        time_array = None
        #print(trajectories)
        
        for trajectory_index, trajectory in enumerate(trajectories) :
            if trajectories_array is None :
                # prepare the array with the trajectories in a format that c_x
                # is expecting; the shape is (n_samples) x (n_initial_conditions) x (n_variables)
                trajectories_array = np.zeros(
                    (
                        len(trajectory["y"][0]), 
                        len(trajectories),
                        len(state_variables)
                    ))
                # also store the values of time, they will be used later
                time_array = np.zeros((len(trajectory["t"]), len(trajectories)))
                
            for variable_index, variable in enumerate(state_variables) :
                trajectories_array[:,trajectory_index,variable_index] = trajectory["y"][variable_index]
            
            time_array[:,trajectory_index] = trajectory["t"]
            
        # at this point, we have the complete trajectory array; so we can
        # finally compute the different 'ingredients' of c_x
        T = time_array[-1,0] # maximum time
        t_new = time_array[:,0]
        
        # get integration weights
        weight = np.ones_like(t_new)
        weight[0] /= 2
        weight[-1] /= 2
        weight = weight / weight.sum() * T
        
        # compute g'(t) and g(t) values using type and number of support functions
        basis_func = basis(T, n_basis)
        g_dot = basis_func.design_matrix(t_new, derivative=True)
        g = basis_func.design_matrix(t_new, derivative=False)
        
        # we should probably start iterating over variables here?
        # all the other parts are not dependant on the index of the variable
        for variable_index, state_variable in enumerate(state_variables) :
            
            # compute c for all trajectories of one specific variable
            Xi = trajectories_array[:, :, variable_index]
            c = (Xi * weight[:, None]).T @ g_dot
            
            # now, we can compute the value of the fitness function c_x for
            # the ground truth equation on all trajectories
            # this code is mostly cut/pasted from D-CODE
            x_hat = trajectories_array
            T, B, D = x_hat.shape
            x_hat_long = x_hat.reshape((T * B, D))
            
            # compute values using the ground truth equation and sympy
            equation = sympy.sympify(equations[state_variable])
            symbols = [sympy.sympify(s) for s in state_variables]
            symbol_values = [x_hat_long[:,i] for i in range(0, len(state_variables))]
            y_hat_long = lambdify(symbols, equation)(*symbol_values)
            
            # finally obtain fitness function
            y_hat = y_hat_long.reshape((T, B))
            c_hat = (y_hat * weight[:, None]).T @ g
            
            # sample_weight was an array of ones, even in the original
            sample_weight = np.ones(c.shape)
            c_x_fitness_value = np.sum((c + c_hat) ** 2 * sample_weight[:, None]) / np.sum(sample_weight)
            
            print("%s -> C_x fitness value for all trajectories: %.4f" %
                  (state_variable, c_x_fitness_value))
        
        print()
        
    return

if __name__ == "__main__" :
    #sys.exit( main_lorenz() )
    sys.exit( main_odebench() )