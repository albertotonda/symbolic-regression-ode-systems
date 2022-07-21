"""
Learn equations using pySR.
"""
import pandas as pd
import sys

from pysr import PySRRegressor

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
            procs=4,
            populations=8,
            population_size=50,
            model_selection="best",  # Result is mix of simplicity+accuracy
            niterations=40,
            binary_operators=["+", "*", "/", "-"],
            unary_operators=[
            "cos",
            "exp",
            "sin",
            ],
        )

        # create training dataset (it can be a DataFrame!)
        y = df[t]
        X = df[features]

        # fit the model
        model.fit(X, y)
        print(model)

        # TODO next steps: - get the model(s), store them in the dictionary
        #                  - set the delta_t to zero and perform the other magic
        #                  - create systems of ODEs


    return results

def main() :
    
    # local imports, used only in the main
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Python script to learn equations that have been modified by the Euler method. By Alberto Tonda, 2022 <alberto.tonda@gmail.com>')
    parser.add_argument("--csv", metavar="dataset.csv", type=str, nargs=1, help="Dataset in CSV format. Must contain a column named \"delta_t\".", required=True)
    args = parser.parse_args()

    # read dataset
    df = pd.read_csv(args.csv[0])

    # function that launces PySR
    equations = learn_equations(df)


    return

if __name__ == "__main__" :
    sys.exit( main() )
