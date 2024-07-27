# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:23:11 2024

Similar to the script "check_equations_in_euler_first_approximation_space.py",
but this time we are going to read a file created by the ODEBench benchmark
and proceed from that.

@author: Alberto
"""
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re as regex
import seaborn as sns
import sympy
import sys
import unicodedata # this is just used to sanitize strings for file names

from sklearn.metrics import r2_score
from sympy.utilities.lambdify import lambdify

# local package(s), imported from another directory
# using a 'parent package' would be cleaner; but for the moment, let's use a brutal way
sys.path.append("../src/")
from explicit_euler_method import apply_euler_method

def slugify(value, allow_unicode=False) :
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = regex.sub(r'[^\w\s-]', '', value.lower())
    return regex.sub(r'[-\s]+', '-', value).strip('-_')

def create_latex_table(results, min_state_variables=1, max_state_variables=1) :
    """
    Utility function to create a latex table with a summary of the results. The
    arguments are used to select the group of results to include. In the original
    conception, I am going to have 3 tables: one for ODE systems with 1 equation, one
    for ODE systems with 2 equations, one for ODE systems with 3-4 equations.
    """
    # set of html colors that will be used to color the cells in the table
    html_colors = {
        'best' : 'ccff00',
        'good' : 'a0d6b4',
        'bad' : 'ee82ee',
        'worst' : 'a74ac7',
        }
    
    latex_table = r'\begin{table}[htb]' + '\n'
    latex_table += r'\centering' + '\n'
    latex_table += r'\resizebox{0.99\textwidth}{!}{%' + '\n'
    #latex_table += r'\begin{tabular}{c|c|c|c|c|c}' + '\n'
    latex_table += r'\begin{tabular}{cccccc}' + '\n' # let's try to go for a nicer style
    latex_table += r'\textbf{Id} & \textbf{Description} & \textbf{State variable} & \textbf{$F$} & \textbf{Trajectory} & \textbf{$R2$}\\' + '\n'
    latex_table += r'\hline \hline' + '\n'
    
    results_index = 0
    while results_index < len(results) :
        
        # local variable, for easier manipulation
        results_system = results[results_index]
        
        # get the keys/names of the state variables
        state_variables = [k for k in results_system if k != "id" and k != "eq_description"]
        # get also the names of the trajectories
        trajectories = [k for k in results_system[state_variables[0]] if k != "F"]
        
        if len(state_variables) >= min_state_variables and len(state_variables) <= max_state_variables :
        
            # this first part will take a number of rows equal to the number of state variables * number of trajectories
            n_rows = len(state_variables) * len(trajectories)
            latex_table += r'\multirow{' + str(n_rows) + '}{*}{' + str(results_system["id"]) + r'} & ' 
            latex_table += r'\multirow{' + str(n_rows) + '}{*}{\parbox{5cm}{' + results_system["eq_description"] + r'}} & '
            
            # now, each cell for a state variable will take a number of rows equal to the number of trajectories
            for state_variable in state_variables :
                
                # here the multirow parameter is the number of trajectories
                n_rows = len(trajectories)
                # add name of the variable and its equation
                latex_table += r'\multirow{' + str(n_rows) + r'}{*}{$' + state_variable + r'$} & '
                # we set the equation to visualize only 3 significant digits for the numerical parameters
                equation_sympy = sympy.sympify(results_system[state_variable]["F"])
                equation_string = sympy.latex(equation_sympy.evalf(4))
                latex_table += r'\multirow{' + str(n_rows) + r'}{*}{$' + equation_string + r'$} & '
                
                # now, the performance of the equation for each trajectory
                for trajectory_index, t in enumerate(trajectories) :
                    
                    # we also add some colors to mark the trajectories for which
                    # the real equation performed well (or not well at all)
                    r2_value = results_system[state_variable][t]
                    performance = 'best'
                    if r2_value < 0.99 and r2_value > 0.9 :
                        performance = 'good'
                    elif r2_value <= 0.9 and r2_value > 0.5 :
                        performance = 'bad'
                    elif r2_value <= 0.5 :
                        performance = 'worst'
                    
                    latex_table += r'\cellcolor[HTML]{' + html_colors[performance] + r'}'
                    latex_table += str(trajectory_index+1) + ' & '
                    latex_table += r'\cellcolor[HTML]{' + html_colors[performance] + r'}'
                    latex_table += "%.6f" % r2_value + r' \\' + '\n'
                    
                    # add an empty row, unless this is the last trajectory
                    if t != trajectories[-1] :
                        #latex_table += '\cline{5-6}' + '\n'
                        latex_table += r' & & & & '
                
                # if this is not the last state variable, add some extra Latex
                if state_variable != state_variables[-1] :
                    #latex_table += r'\cline{3-6}' + '\n'
                    latex_table += r' & & '
        
            # add a line in the table
            latex_table += r'\hline' + '\n'
        
        results_index += 1
        
    # end of the table
    latex_table += r'\end{tabular}%' + '\n'
    latex_table += r'}' + '\n' # end of the resizebox
    latex_table += r'\end{table}' + '\n'
    
    return latex_table

def main() :
    
    # some hard-coded values
    # source file, in JSON
    odebench_file_name = "../data/odebench/all_odebench_trajectories.json"
    #odebench_file_name = "../data/odebench/selected_equations_600_points_trajectories.json"
    # this one is another file that allegedly should contain the same information,
    # but it is in fact different (!)
    #odebench_file_name = "../local_files/odeformer/odeformer/odebench/strogatz_extended.json"
    # results folder
    results_directory = "../local_results/checking-odebench"
    #results_directory = "../local_results/checking-odebench-selected-600-points"
    
    # this data structure will be used to collect the results
    results = []
    
    # set a nice style for plots
    sns.set_style('darkgrid')
    
    if not os.path.exists(results_directory) :
        os.makedirs(results_directory)
    
    # read the file
    odebench = None
    with open(odebench_file_name, "r") as fp :
        odebench = json.load(fp)
        
    # for some reason, the JSON file is a list of dictionaries, so let's go with the flow
    for system in odebench :
        print("Working on system %d: \"%s\"" % (system["id"], system["eq_description"]))
        
        # let's create a subdirectory for this system
        system_directory = slugify("%d-%s" % (system["id"], system["eq_description"]))
        system_directory = os.path.join(results_directory, system_directory)
        if not os.path.exists(system_directory) :
            os.makedirs(system_directory)
        
        # get the description of the state variables, and capture their names
        state_variables = regex.findall("([0-9|a-z|\_]+)\:\s+", system["var_description"])
        #print("State variables:", state_variables)
        
        # start preparing the sub-dictionary for the results
        results_system = {s : {} for s in state_variables}
        results_system["id"] = system["id"]
        results_system["eq_description"] = system["eq_description"]
        
        # associate each variable with the expression of its derivative; for
        # some weird reason, the key "substituted" returns a list of ONE element (!)
        # that contains a list of strings ^_^;;
        equations = {state_variables[i] : sympy.sympify(system["substituted"][0][i])
                     for i in range(0, len(state_variables))}
        
        # now, we can proceed to read the data (integrated trajectory/ies); there
        # are in fact TWO distinct trajectories with different initial conditions (!)
        trajectories = system["solutions"][0] # for some reason, another list with 1 element (...) 
        print("Found a total of %d trajectories!" % len(trajectories))
        
        # let's go trajectory by trajectory
        for trajectory_index, trajectory in enumerate(trajectories) :
            
            # now, the weird thing about the data structure is that values for
            # 't' are presented as a key in the dictionary; while all the others
            # are presented as items in a list under key 'y' (...)
            dictionary_trajectory = {}
            dictionary_trajectory["t"] = trajectory["t"]
            
            for v in range(0, len(state_variables)) :
                dictionary_trajectory[state_variables[v]] = trajectory["y"][v]
            
            print("- Trajectory #%d contains %d points" % (trajectory_index, len(dictionary_trajectory["t"])))
        
            # convert the dictionary with the values to a pandas DataFrame,
            # and save it (why not?)
            df_trajectory = pd.DataFrame.from_dict(dictionary_trajectory)
            df_trajectory.to_csv(os.path.join(system_directory, "trajectory-%d.csv" % trajectory_index), index=False)
            
            # let's plot the trajectories!
            for state_variable in state_variables :
                
                fig, ax = plt.subplots(figsize=(10,8))
                ax.scatter(df_trajectory["t"].values, df_trajectory[state_variable].values, alpha=0.7)
                ax.set_title("Trajectory %d, state variable \"$%s$\"" % (trajectory_index, state_variable))
                ax.set_xlabel("t")
                ax.set_ylabel("$" + state_variable + "$")
                fig.tight_layout()
                
                figure_file_name = "trajectory-%d-%s.png" % (trajectory_index, state_variable)
                plt.savefig(os.path.join(system_directory, figure_file_name), dpi=150)
                plt.close(fig)
                
            # now, we need to check that the known ground truth equations for
            # the F_y functions actually have a good fit; let's first transform
            # the data set into the forward Euler's method feature space
            df_euler = apply_euler_method(df_trajectory)
            df_euler.to_csv(os.path.join(system_directory, "trajectory-%d-euler.csv" % trajectory_index), index=False)
            
            # for each state variable y, obtain the F_y known ground truth form
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
                #equation_symbols = [sympy.sympify(s) for s in df_euler.columns]
                symbol_values = [df_euler[c].values for c in df_euler.columns]
                
                # obtain predicted values
                equation_values = lambdify(equation_symbols, euler_equation)(*symbol_values)
                ground_truth = df_euler["F_" + state_variable].values
                
                # compute R2 score
                r2_value = r2_score(ground_truth, equation_values)
                
                # some printouts
                print("-- For state variable \"%s\", real F_%s has R2=%.6f" %
                      (state_variable, state_variable, r2_value))
                
                # now, it is interesting to actually plot the results to
                # check where the residuals are the largest (e.g. where the
                # approximation fails with respect to the actual values)
                # TODO but this is the derivative! maybe we can do something better
                residuals = abs(ground_truth - equation_values)
                
                fig, ax = plt.subplots(figsize=(10,8))
                scatter = ax.scatter(df_euler["t"].values, ground_truth, c=residuals,
                           marker='.', cmap='plasma', alpha=0.7, 
                           label="Dynamic for variable %s" % state_variable)
                cbar = plt.colorbar(scatter)
                cbar.set_label("Values of the residuals")
                
                ax.set_xlabel("t")
                ax.set_ylabel("$" + state_variable + "$")
                ax.set_title("Dynamic for variable $%s$, color-coded by values of residuals" 
                             % state_variable)
                
                plt.savefig(os.path.join(system_directory, 
                                         "trajectory-%d-residuals-%s.png" % 
                                         (trajectory_index, state_variable)), 
                            dpi=150)
                plt.close(fig)
                
                # save results in the dictionary
                results_system[state_variable]["F"] = str(euler_equation)
                results_system[state_variable]["trajectory-%d" % trajectory_index] = r2_value
                
        #sys.exit(0) # TODO remove this, it's just for debugging
        # here is the end of the loop for a system
        results.append(results_system)
        
    # it would be nice to create a Latex table starting from the results
    # collected in the "results" list; since there are 63 (!) systems, maybe
    # I could copy what the people in ODEBench did, and just go for separate
    # tables, depending on the number of equations in the system
    
    # now, the function here below works; but it would be better to actually
    # separate this script from the Latex post-processing, this is a TODO
    
    # use the function to create different tables, for 1, 2, and 3-4 state variables
    print("Now postprocessing results...")
    latex_table_1 = create_latex_table(results, min_state_variables=1, max_state_variables=1)
    # write table to file
    with open(os.path.join(results_directory, "table-1-state-variable.tex"), "w") as fp :
        fp.write(latex_table_1)
        
    latex_table_2 = create_latex_table(results, min_state_variables=2, max_state_variables=2)
    with open(os.path.join(results_directory, "table-2-state-variables.tex"), "w") as fp :
        fp.write(latex_table_2)
        
    latex_table_3_4 = create_latex_table(results, min_state_variables=3, max_state_variables=4)
    with open(os.path.join(results_directory, "table-3-4-state-variables.tex"), "w") as fp :
        fp.write(latex_table_3_4)
    
    return

if __name__ == "__main__" :
    sys.exit(main())