# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:05:52 2025

This script is trying to validate the C_x approach, using their original data,
but also attempting to re-implement the evaluation in a more convenient external
python function.

@author: Alberto
"""
import numpy as np
import os
import pickle
import sympy
import sys

from sympy.utilities.lambdify import lambdify

# brutally add the main directory of D-CODE to the python path
sys.path.append("../../D-CODE-ICLR-2022")

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
    equation = sympy.sympify(ground_truth_x0)
    
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
    # an ODEBench trajectory

if __name__ == "__main__" :
    sys.exit( main_lorenz() )