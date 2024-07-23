# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:34:21 2024

This script is just to check whether the equations converted to Euler's first
approximation's space really represent the best possible fit for the numerical
procedure that numerically transforms the data into the new space.

@author: Alberto
"""
import matplotlib.pyplot as plt
import os
import seaborn as sns
import shutil
import sympy
import sys

from sklearn.metrics import r2_score, PredictionErrorDisplay
from sympy.utilities.lambdify import lambdify

# local package
from create_dataset_from_ode_system import get_df_from_ode_system, parse_ode_from_text
from explicit_euler_method import apply_euler_method

def main() :
    
    # pick the text file of the system here
    #ode_system_file_name = "../data/lotka-volterra.txt"
    ode_system_file_name = "../data/rossler-stable.txt"
    
    # we will be plotting stuff, so let's set a nice style
    sns.set_style('darkgrid')
    
    # create directory based on the name of the original ODE system file
    ode_system_name = os.path.basename(ode_system_file_name)[:-4]
    results_directory = "checking-" + ode_system_name
    
    if not os.path.exists(results_directory) :
        os.makedirs(results_directory)
    
    # copy the original file inside the folder, if it is not there yet
    destination_ode_system_file_name = os.path.join(results_directory, os.path.basename(ode_system_file_name))
    if not os.path.exists(destination_ode_system_file_name) :
        shutil.copyfile(ode_system_file_name, destination_ode_system_file_name)    
    
    # get the system from the file, solve it for the initial conditions given in
    # the text file and create a data set
    equations, ordered_variables, initial_conditions, time_step, max_time = parse_ode_from_text(ode_system_file_name)
    df_ode = get_df_from_ode_system(equations, ordered_variables, initial_conditions, time_step, max_time)
    # save the data set to file
    df_ode.to_csv(os.path.join(results_directory, ode_system_name + ".csv"), index=False)
    
    # now, convert the data set to Euler's first approximation's space
    df_euler = apply_euler_method(df_ode)
    df_euler.to_csv(os.path.join(results_directory, ode_system_name + "-euler.csv"), index=False)
    
    # now, it's actually quite easy to convert the original equations to their
    # version in Euler's first approximation's space; just multiply everything
    # by the new symbol 'delta_t'
    delta_t = sympy.Symbol("delta_t")
    euler_equations = { variable : sympy.Mul(delta_t, equation)  
                       for variable, equation in equations.items() }
    print("Original equations:", equations)
    print("Equations in Euler's first approximation's space:", euler_equations)
    
    # and now, let's see if they fit! we can easily lambdify symbolic expressions 
    # in sympy, and then just use them as numpy functions
    equation_symbols = [sympy.sympify(s) for s in df_euler.columns]
    symbol_values = [df_euler[c].values for c in df_euler.columns]
    
    for variable, equation in euler_equations.items() :
        equation_values = lambdify(equation_symbols, equation)(*symbol_values) # flatten dictionary, use as arguments
        ground_truth = df_euler["F_" + variable].values
        r2_value = r2_score(ground_truth, equation_values)
        
        print("%s (%s): R2=%.10f" % (str(variable), str(equation), r2_value))
        
        fig, ax = plt.subplots(figsize=(10,8))
        display = PredictionErrorDisplay.from_predictions(ground_truth, equation_values, 
                                                          kind="actual_vs_predicted", ax=ax)
        ax.set_title("State variable \"%s\" (%s), R2=%.10f" % (str(variable), str(equation), r2_value))
        plt.savefig(os.path.join(results_directory, "predictions-" + str(variable) + ".png"), dpi=300)
        plt.close(fig)
        
        # add extra columns to the Euler data set, with predictions and residuals
        df_euler["F_" + variable + "_pred"] = equation_values
        df_euler["F_" + variable + "_residuals"] = abs(ground_truth - equation_values)
        
    # it is also be interesting to find the 10 largest errors, and see
    # for which combination of values they appear; for the moment, I suspect that
    # the largest errors will be for points where F_x(t_n, x_n, delta_t) == F_x(t_n, x_n, 0)
    
    # overwrite the Euler data set with its current content
    df_euler.to_csv(os.path.join(results_directory, ode_system_name + "-euler.csv"), index=False)
    
    # now, for each state variable, sort the data frame by residual size and try to
    # find a relationship between the largest residuals and the values of the variables
    for variable in euler_equations :
        residuals_column = "F_" + variable + "_residuals"
        
        # sort data frame on that column, largest to smallest
        df_euler_sorted = df_euler.sort_values(by=residuals_column, ascending=False)
        
        # start by saving a separate copy of the sorted data frame, keeping
        # only the interesting columns
        interesting_columns = [c for c in df_euler_sorted.columns
                               if not c.startswith("F_") 
                               or c.startswith("F_" + variable)]
        
        df_euler_sorted[interesting_columns].to_csv(
            os.path.join(results_directory, "residuals-" + variable + ".csv"), index=False)
        
        # another interesting thing we can do is to plot the dynamic for the
        # variable, BUT color-coded with a heatmap that shows where the residuals
        # are the highest; looking at several examples, it seems that F_x(t_n, y_n, 0)
        # always has residuals equal to zero (makes sense, since delta_t == 0),
        # so we can just focus on residuals for F_x(t_n, y_n, delta_t)
        df_plot = df_euler[df_euler["delta_t"] != 0]
        
        fig, ax = plt.subplots(figsize=(10,8))
        scatter = ax.scatter(df_plot["t"].values, df_plot[variable].values, c=df_plot[residuals_column].values,
                   cmap='plasma', label="Dynamic for variable %s" % variable)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Values of the residuals")
        
        ax.set_xlabel("t")
        ax.set_ylabel(variable)
        ax.set_title("Dynamic for variable %s, color-coded by values of residuals" % variable)
        
        plt.savefig(os.path.join(results_directory, "residuals-" + variable + ".png"), dpi=300)
        plt.close(fig)
        
        
    return


if __name__ == "__main__" :
    sys.exit( main() )
