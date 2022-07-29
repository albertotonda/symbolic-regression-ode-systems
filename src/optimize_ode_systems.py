"""
Functions to replace fixed values with parameters, and optimize them, in ODE systems.
"""
import cma
import numpy
import sys
import sympy

from scipy import integrate

def parametrize_expressions(expressions) :
    """
    Given a dictionary of sympy expression, replace all numerical values with symbolic parameters in the form 'p_n'
    """

    n_parameters = 0
    initial_values = {}
    parametrized_expressions = {}

    for variable, expression in expressions.items() :
        for s in expression.atoms(sympy.Number) :
            
            parameter = sympy.symbols("p_%d" % n_parameters)
            expression = expression.subs(s, parameter)
            initial_values[parameter] = s.evalf()

            n_parameters += 1

        parametrized_expressions[variable] = expression

    return parametrized_expressions, initial_values

def optimize_ode_parameters(equations, initial_values, df_train) :
    """
    Optimize parameters of an ODE system using CMA-ES. We could start from the assumption that the actual values will not be too far away from the
    parameters found by the symbolic regression, to limit the search space.
    """

    # get variables, in order
    variables = [v for v, e in equations.items()]

    # get initial conditions for solving the differential equation
    initial_conditions = [df_train[v].values[0] for v in variables]

    # get array of time values
    t = df_train["t"].values

    # get all initial values of parameters in the expressions; as of Python 3.7, dictionaries preserve the order of insertion of keys
    # so we can just iterate and store values for the starting point of the search
    x_0 = [v for p, v in initial_values.items()]
    parameter_names = [p for p, v in initial_values.items()]

    # initialize CMA-ES optimization algorithm
    es = cma.CMAEvolutionStrategy(x_0, 1e-2, {'popsize': 100})

    # the fitness function takes the symbolic expressions, replaces
    # the symbolic parameters with the (candidate optimal) values,
    # and solves the resulting system as an ODE system
    es.optimize(fitness_function, args=(parameter_names, equations, variables, df_train))
    best_parameter_values = es.result[0]

    return best_parameter_values

def dX_dt(X_local, t_local, equations : dict, symbols : list) :
    """
    Function that is used to solve the differential equation 
    """
    # create dictionary symbol -> value (order of symbols is important!)
    symbols_to_values = {s: X_local[i] for i, s in enumerate(symbols)}
    symbols_to_values['t'] = t_local

    # compute values, evaluating the symbolic functions after replacement
    values = [ eq.evalf(subs=symbols_to_values) for var, eq in equations.items() ]

    return values

def fitness_function(parameter_values, parameter_names, equations : dict, variables : list, df) :
    """
    Computes the mean squared error between the data and the system solved with the parameters.
    """
    # create dictionary to later replace parameters with their values
    replacement_dictionary = { parameter_names[i] : parameter_values[i] for i in range(len(parameter_names)) }

    # replace symbolic parameters with their local (candidate) values
    local_equations = {}
    for var, eq in equations.items() :
        local_equations[var] = equations[var].subs(replacement_dictionary)

    # solve the system for the initial conditions 
    initial_conditions = [ df[v].values[0] for v in variables ]
    t = df["t"].values
    Y, info_dict = integrate.odeint(dX_dt, initial_conditions, t, args=(local_equations, variables, ), full_output=True)

    # TODO some checks in case of errors or horrible values
    
    # compute the mean squared error
    mse = ((Y - df[variables].values)**2).mean(axis=0).mean(axis=0)

    return mse

def main() :

    import os
    import pandas as pd
    import re

    from sympy.parsing.sympy_parser import parse_expr

    # get the list of all systems of equations in the target folder
    target_folder = "2022-07-21-16-45-08-lotka-volterra"
    files_list = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.startswith("candidate") and f.endswith(".txt")]
    print("Found a total of %d candidate ODE systems" % len(files_list))

    # read the data (TODO split data into training and test?)
    df = pd.read_csv(os.path.join(target_folder, "lotka-volterra.csv"))

    for i, file_system in enumerate(files_list) :

        # get an array of sympy equations
        lines = []
        equations = {}
        with open(file_system, "r") as fp : lines = fp.readlines()

        for line in lines :
            tokens = line.split("=")
            # get the name of the variable with a regular expression
            match = re.search("d([\w]+)/dt", tokens[0])
            equations[match.group(1)] = parse_expr(tokens[1])

        print(equations)

        # let's try to integrate the function
        variables = [ v for v, e in equations.items() ]
        initial_conditions = [ df[v].values[0] for v in variables ]
        t = df["t"].values
        Y, info_dict = integrate.odeint(dX_dt, initial_conditions, t, args=(equations, variables, ), full_output=True)

        # plot(s)? just to see what happens
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # measured values
        ax.plot(df[variables[0]].values, df[variables[1]].values, 'g-', label="Original values")
        ax.plot(Y[:,0], Y[:,1], 'r-', label="Predicted values")

        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_title("ODE system %d" % i)
        ax.legend(loc='best')

        plt.savefig(file_system[:-4] + ".png", dpi=300)
        plt.close(fig)

        for i, v in enumerate(variables) :
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(t, Y[:,i], 'r-', label="Predicted values")
            ax.plot(t, df[v].values, 'g-', label="Original values")

            plt.savefig(file_system[:-4] + ".png", dpi=300)
            plt.close(fig)
        
        # parametrize functions
        parametrized_expressions, initial_values = parametrize_expressions(equations)
        print(parametrized_expressions)
        print(initial_values)
        
        mse = ((Y - df[variables].values)**2).mean(axis=0).mean(axis=0)
        print("MSE:", mse)

        # optimize parameters
        print("Now optimizing parameters...")
        best_parameter_values = optimize_ode_parameters(parametrized_expressions, initial_values, df)

        sys.exit(0)

    return

if __name__ == "__main__" :
    sys.exit( main() )
