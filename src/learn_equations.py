"""
Learn equations using pySR.
"""
import pandas as pd
import sys

from pysr import PySRRegressor
from sympy import diff, simplify, symbols
from sympy.parsing.sympy_parser import parse_expr

def learn_equations(df) :

    # analyze the columns, divide them into targets and features
    targets = [ c for c in df.columns if c.startswith("F_") ]
    features = [ c for c in df.columns if c not in targets ]

    # dictionary of results, at the end it will contain variable -> list of symbolic equations
    results = { f : [] for f in features }

    # go over each target, and start PySR
    for t in targets :

        print("Starting regression for \"%s\"..." % t)

        # instantiate pySR regressor
        model = PySRRegressor(
            procs=7,
            populations=8,
            population_size=50,
            batching=True, # use batches instead of the whole dataset
            batch_size=50, # 50 is the default value for the batches
            model_selection="best",  # Result is mix of simplicity+accuracy
            niterations=80,
            binary_operators=["+", "*", "/", "-"],
            unary_operators=[
            "cos",
            "exp",
            "sin",
            ],
        )

        # create training dataset (it can be a DataFrame, which also removes the necessity of specifying the variable names!)
        y = df[t]
        X = df[features]

        # fit the model
        model.fit(X, y)
        print(model)

        # get the final Pareto front of equations (in symbolic form) and save them into the dictionary
        # equations_ here are not in symbolic form, they are strings, so they need to be re-converted
        results[t] = [ parse_expr(eq) for eq in model.equations_["equation"] ]
        print(results[t])

    return results

def prune_equations(equations) :
    """
    Taking in input an iterable (list or dataframe, probably) of symbolic equations, performs derivation for delta_t, 
    set delta_t to zero and then removes all duplicates and all equations that are equal to zero.
    """

    equations_replaced = []

    # create symbol for "delta_t"
    delta_t = symbols("delta_t") 

    # derivate for delta_t, then set remaining delta_t to zero
    for eq in equations :
       
        # derivate
        eq_diff = diff(eq, delta_t) 

        # replace any remaining delta_t with 0
        equations_replaced.append( eq_diff.subs(delta_t, 0) )

    # prune/remove all duplicates and equations that reduce to constants
    # TODO  to further reduce the number of equations, I could try to replace numerical parameters with variables,
    #       but then, it's still hard to find stuff.
    pruned_equations = []
    for i_1, eq_1 in enumerate(equations_replaced) :

        # first, check if the simplified equation contains only constants
        if not simplify(eq_1).is_constant() :

            # then, check if the equation is a duplicate of something we
            # already have stored among the pruned equations
            is_equation_duplicate = False
        
            for i_2, eq_2 in enumerate(pruned_equations) :
                if simplify(eq_2-eq_1) == 0 : # this happens only if the two equations are exactly the same (symbolically)
                    is_equation_duplicate = True

            if is_equation_duplicate == False :
                pruned_equations.append(eq_1)

    return pruned_equations

def main() :

    if False :
        # this is just brutal testing for pruning
        df = pd.read_csv("hall_of_fame_2022-07-22_141320.949.csv", sep="|")
        print(df)

        pruned_equations = prune_equations([ parse_expr(eq) for eq in df["Equation"].values ])
        print("Equations after pruning:")
        print(pruned_equations)

    if True :
        # local imports, used only in the main
        import argparse
        import os

        parser = argparse.ArgumentParser(description='Python script to learn equations that have been modified by the Euler method. By Alberto Tonda, 2022 <alberto.tonda@gmail.com>')
        parser.add_argument("--csv", metavar="dataset.csv", type=str, nargs=1, help="Dataset in CSV format. Must contain a column named \"delta_t\".", required=True)
        args = parser.parse_args()

        # read dataset
        df = pd.read_csv(args.csv[0])

        # function that launches PySR and returns a dictionary of equations for each variable
        equations = learn_equations(df)
        print(equations)

        # new dictionary with the pruned equations
        pruned_equations = dict()
        for var, eqs in equations.items() :
            if var.startswith("F_") :
                pruned_equations["d" + var[2:] + "/dt"] = prune_equations(eqs)
        print("Pruned equations:", pruned_equations)

        # build the systems of differential equations, using the 'itertools' package
        # but before that, we will need to create a list of lists (TODO and store the fact that we are deriving?)
        equations_list = [ eqs for var, eqs in pruned_equations.items() ]
        variables_list = [ var for var, eqs in pruned_equations.items() ]

        import itertools
        ode_systems = list(itertools.product(*equations_list))

        print("Resulting candidate ODE systems:")
        print(ode_systems)

        print("Saving ODE systems to disk:")
        # get path of the original CSV dataset
        original_path, original_csv = os.path.split(args.csv[0]) 

        for index, system in enumerate(ode_systems) :

            text = ""
            for i in range(0, len(system)) :
                text += str(variables_list[i]) + " = "
                text += str(system[i]) + "\n"

            with open(os.path.join(original_path, "candidate_system_%d.txt" % index), "w") as fp :
                fp.write(text)

    return

if __name__ == "__main__" :
    sys.exit( main() )
