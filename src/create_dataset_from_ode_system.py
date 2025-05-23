"""
This script parses text or JSON information with an ODE system and
some initial conditions, solves the system and saves the data as a
CSV. There is no error control, so input data in the wrong format
will make it crash.
"""
import importlib
import json
import numpy as np
import os
import pandas as pd
import re
import sympy
import sys
import time

from scipy import integrate

def parse_ode_from_json(file_name) :

    # read the JSON file
    dictionary = dict()
    with open(file_name, "r") as fp : 
        dictionary = json.load(fp)

    # some data structures that will be used later
    equations = dict()
    initial_conditions = dict()
    ordered_variables = []

    # parse the equations in the ODE system
    for variable, equation in dictionary["ode_system"].items() :

            matches = re.match('d([A-Za-z0-9_]+)/dt', variable)
            variable_name = matches.group(1)
            equations[variable_name] = sympy.sympify(equation)

            ordered_variables.append(variable_name)

    # parse initial conditions
    for variable, initial_condition in dictionary["initial_conditions"].items() :
        initial_conditions[variable] = initial_condition

    # and final details
    return equations, ordered_variables, initial_conditions, dictionary["time_step"], dictionary["max_time"]

def parse_ode_from_text(file_name) :

    lines = []
    with open(file_name, "r") as fp : lines = fp.readlines()

    # some data structures that will be used later
    equations = dict()
    initial_conditions = dict()
    ordered_variables = []

    # let's perform a line-by-line analysis of the text file
    for line in lines :
        line = line.strip() # remove whitespaces and stuff

        # if the line starts with 'd', it's a differential equation
        if line[0] == 'd' :

            # get the name of the variable (e.g. dx/dt -> 'x')
            matches = re.match('d([A-Za-z0-9_]+)/dt', line)
            variable_name = matches.group(1)

            # tokenize the line on '=' to get the equation
            # (e.g. "dx/dt = x + 5")
            tokens = line.split('=')

            # associate the symbolic equation to the variable name in the dictionary
            equations[variable_name] = sympy.sympify(tokens[1])
            # also, take note of the order the variables appear in
            ordered_variables.append(variable_name)

        # if the line starts with 'i', it's part of the initial conditions
        elif line[0] == 'i' :

            # split the line on ":"
            tokens = line.split(':')

            # now, let's use a regex to match all the "X_1 = x1" groups
            matches = re.findall(r'([A-Za-z0-9_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)', tokens[1])

            for match in matches :
                initial_conditions[match[0]] = float(match[1])

        # line starting with 't' is the time step
        elif line[0] == 't' :
            tokens = line.split(':')
            time_step = float(tokens[1])

        # line starting with 'm', it's the maximum time
        elif line[0] == 'm' :
            tokens = line.split(':')
            max_time = float(tokens[1])

    return equations, ordered_variables, initial_conditions, time_step, max_time

def write_temp_ode_system(equations, ordered_variables, file_name) :

    indentation = "    "
    text = ""
    

    text += "# This Python script has been automatically generated\n"
    text += "# On " + time.strftime("%Y-%m-%d, %H:%M:%S") + "\n\n"

    text += "import os\nimport sys\n\nfrom math import *\n\n"
    
    text += "def dX_dt(t, X) :\n"
    for i in range(0, len(ordered_variables)) :
        text += indentation + str(ordered_variables[i]) + " = X[" + str(i) + "]\n"
    text += indentation + "return_values = []\n\n"
    for key in ordered_variables :
        text += indentation + "return_values.append( " + str(equations[key]) + " )\n"

    text += "\n" + indentation + "return return_values"

    with open(file_name, "w") as fp : fp.write(text)

    return 

def solve_ode_system(ordered_variables, initial_conditions, time_step, max_time, module_name) :
    
    # dinamically import the temporary module (hopefully) containing the ODE system
    ode = importlib.import_module(module_name)

    # total number of values that we will obtain through integration
    size = int(max_time / time_step + 1)
    Y = np.zeros((size, len(ordered_variables))) # output
    time = np.zeros(size) # time array

    # setup, storing values of the initial conditions for time=0.0
    for i in range(0, len(ordered_variables)) :
        Y[0,i] = initial_conditions[ordered_variables[i]]
    time[0] = 0.0
    
    r = integrate.ode(ode.dX_dt).set_integrator('dopri5')
    r.set_initial_value(Y[0], time[0])

    index = 1
    while r.successful() and r.t < max_time :
        r.integrate(r.t + time_step, step=True)
        Y[index,:] = r.y
        time[index] = r.t

        index += 1 

    return Y, time

def get_df_from_ode_system(equations, ordered_variables, initial_conditions, time_step, max_time) :
    """
    Gets data integrating an ODE system given the system in symbolic form, 
    initial conditions, time step, and all that jazz.
    """
    temp_file_name = "temp.py"
    
    # unfortunately I still have not found a better way to perform the integration
    # so I need to write a temporary file, import it, run it and then delete it
    write_temp_ode_system(equations, ordered_variables, temp_file_name)
    
    # solve the system get the data
    Y, t = solve_ode_system(ordered_variables, initial_conditions, time_step, max_time, temp_file_name[:-3])
    
    # delete temporary file
    if os.path.exists(temp_file_name) :
        os.remove(temp_file_name)
    
    # store data inside a dictionary
    df_dictionary = {ordered_variables[j] : Y[:,j] for j in range(0, len(ordered_variables))}
    df_dictionary["t"] = t
    
    # convert the dictionary with the data to a pandas DataFrame object
    df = pd.DataFrame.from_dict(df_dictionary)
    
    return df

def main() :

    # these packages are only used in the main, so they are imported here, just
    # in case we will later import the script for its functions
    import argparse
    import os
    import pandas as pd

    # parse arguments
    parser = argparse.ArgumentParser(description='Python script to create dataset files, starting from a text description of a dynamic system + initial conditions. By Alberto Tonda, 2022 <alberto.tonda@inrae.fr>')
    parser.add_argument("--system", metavar="system.txt", type=str, nargs=1, 
                        help="input file; format:\ndX_1/dt = <equation>.\ndX_2/dt = <equation>\n...\ndX_n/dt = <equation>\ninitial_conditions: X_1=x1 X_2=x2 ... X_n=xN\ntime_step: 0.1 <optional>i\nmax_time: 100", 
                        required=False)
    args = parser.parse_args()
    
    if args.system is None :
        args.system = ["../data/lotka-volterra.json"]
        print("--system option should be specified on command line. Proceeding with default (\"%s\") value for debugging..." %
              args.system[0])


    # get the new file name
    path, ode_file_name = os.path.split(args.system[0])
    dataset_file_name = ode_file_name.split(".")[0] + "-dataset.csv"

    print("I am going to read file \"%s\" and write to file \"%s\"..." % (ode_file_name, dataset_file_name))

    # check if the file ends with ".txt" or ".json"
    if args.system[0].endswith(".txt") :
        equations, ordered_variables, initial_conditions, time_step, max_time = parse_ode_from_text(args.system[0])

    elif args.system[0].endswith(".json") :
        equations, ordered_variables, initial_conditions, time_step, max_time = parse_ode_from_json(args.system[0])

    else :
        print("The file for the ODE system should be in either .txt or .json format. Aborting...")
        sys.exit(0)


    # debug lines
    print("Equations:", equations)
    print("Initial conditions:", initial_conditions)
    print("time_step=%.4f, max_time=%.4f" % (time_step, max_time))

    # write to file
    write_temp_ode_system(equations, ordered_variables, "temp.py")

    # solve the ODE system and write dataset to file
    Y, t = solve_ode_system(ordered_variables, initial_conditions, time_step, max_time, "temp")

    print("Y is", len(Y))

    # save the result in a nice folder with some stuff
    ode_system_name = ode_file_name.split(".")[0]
    output_folder = time.strftime("%Y-%m-%d-%H-%M-%S-" + ode_system_name)
    os.mkdir(output_folder)  

    # create a dictionary, then a dataframe, it's easier to manage this way
    df_dictionary = dict()
    df_dictionary["t"] = t
    for j in range(0, len(ordered_variables)) :
        df_dictionary[ordered_variables[j]] = [ Y[i][j] for i in range(0, len(Y)) ]

    df = pd.DataFrame.from_dict(df_dictionary)
    df.to_csv(os.path.join(output_folder, ode_system_name + ".csv"), index=False)

    # some plots! (using seaborn for aesthetics)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # all variables vs time
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, v in enumerate(ordered_variables) :
        ax.plot(t, df[v], label=v)

    ax.set_xlabel("Time")
    ax.set_title(ode_system_name)
    ax.legend(loc='best')

    plt.savefig(os.path.join(output_folder, ode_system_name + ".png"), dpi=300) 
    plt.close(fig)

    # some more plots! if we have two variables, we can try a 2D plot
    if len(ordered_variables) == 2 :

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(df[ordered_variables[0]], df[ordered_variables[1]])

        ax.set_title(ode_system_name)
        ax.set_xlabel(ordered_variables[0])
        ax.set_ylabel(ordered_variables[1])

        plt.savefig(os.path.join(output_folder, ode_system_name + "-2D-plot.png"), dpi=300) 
        plt.close(fig)

    # otherwise, we could go for a 3D plot
    elif len(ordered_variables) == 3 :
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        plt.plot(df[ordered_variables[0]], df[ordered_variables[1]], df[ordered_variables[2]])

        plt.savefig(os.path.join(output_folder, ode_system_name + "-3D-plot.png"), dpi=300) 
        plt.close(fig)


    return

if __name__ == "__main__" :
    
    # normally I should call the main here, but let's try some debugging
    #sys.exit( main() )
    
    ode_system_file_name = "../data/lotka-volterra.txt"
    equations, ordered_variables, initial_conditions, time_step, max_time = parse_ode_from_text(ode_system_file_name)
    df = get_df_from_ode_system(equations, ordered_variables, initial_conditions, time_step, max_time)
    
    df.to_csv("lotka-volterra.csv", index=False)
