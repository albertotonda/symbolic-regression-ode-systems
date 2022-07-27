"""
Functions to replace fixed values with parameters, and optimize them, in ODE systems.
"""
import sys
import sympy

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

def optimize_ode_parameters(expressions, initial_values, df_train) :
    """
    Optimize parameters of an ODE system using CMA-ES
    """

    return best_values

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
        
        # parametrize functions
        parametrized_expressions, initial_values = parametrize_expressions(equations)
        print(parametrized_expressions)
        print(initial_values)

        # optimize parameters


        sys.exit(0)

    return

if __name__ == "__main__" :
    sys.exit( main() )
