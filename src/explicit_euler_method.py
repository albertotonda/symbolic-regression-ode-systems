"""
Given a dataset created by a system of Ordinary Differential Equations, apply the (Explicit) Euler Method to
convert it into a dataset of independent regular equations.
"""
import pandas as pd
import sys

# TODO: in theory, delta_t could be bigger than 1, but the original code does not seem to change anything, as it takes the next_time from the following row in the dataset, so delta_t seems to always be exactly 1
def apply_euler_method(df, delta_t=1) :
    """
    Given a dataframe produced by an ODE system, return a modified dataframe applying the Euler Method.
    """

    # we extract values for column 't', as they will be used a lot
    # in the following; NOTE: we could also check here if the column
    # exists, and return an error if it does not
    t = df["t"].values

    # create the new columns names for the dictionary that will later
    # be converted into a dataframe; for each column 'c', we add a 'F_c'
    dict_euler = { "F_" + c : [] for c in df.columns if c != "t"}
    for c in df.columns :
        dict_euler[c] = []
    dict_euler["delta_t"] = []

    # iterate over the dataset, filling the dictionary
    # NOTE: it could be sped up using np arrays, I could evaluate
    # the size of the arrays before creating them, but at the moment
    # it's not really a blocking issue
    for index in range(0, df.shape[0]) :

        # set time values that we will need to compute the Euler method
        next_time = 0
        if index < df.shape[0] - 1 :
            next_time = t[index+1]

        start_time = current_time = t[index]

        # let's look at the following rows
        index_2 = index
        while current_time < next_time and index_2 - index <= delta_t :

            current_time = t[index_2]
            
            # copy values from the columns in the original dataset;
            # also compute values for new variables "F_*"
            for c in df.columns :
                dict_euler[c].append(df[c].values[index])
                if c != "t" :
                    dict_euler["F_" + c].append(df[c].values[index_2] - df[c].values[index])

            # add value of delta_t
            dict_euler["delta_t"].append(current_time - start_time)

            # update index
            index_2 += 1

    df_euler = pd.DataFrame.from_dict(dict_euler)

    return df_euler

def apply_improved_euler_method(df, delta_t=1) :
    """
    The apply_euler_method function does not seem to work properly for values of
    delta_t > 1 (basically it always returns the same result, no matter the delta_t);
    in order to avoid breaking working code, here is a variant that tries to fix
    the issue.
    """
    # TODO: we could still check whether the 't' column exists, and raise an
    # Exception if it doesn't
    
    # create the new columns names for the dictionary that will later
    # be converted into a dataframe; for each column 'c', we add a 'F_c'
    dict_euler = { "F_" + c : [] for c in df.columns if c != "t"}
    for c in df.columns :
        dict_euler[c] = []
    dict_euler["delta_t"] = []

    # iterate over the dataset, filling the dictionary
    for starting_index in range(0, df.shape[0]) :
        # the starting_index is the reference point; the end_index is the "following"
        # reference time step that will be used to compute the time and value differences
        for end_index in range(starting_index, min(starting_index+delta_t+1, df.shape[0])) :
            # copy values from the original columns of the data set; also compute
            # the new values for the F_* functions, for each variable except 't'
            for c in df.columns :
                dict_euler[c].append(df[c].values[starting_index])
                if c != "t" :
                    dict_euler["F_" + c].append(df[c].values[end_index] - df[c].values[starting_index])
            
            # add value for delta_t
            dict_euler["delta_t"].append(df["t"].values[end_index] - df["t"].values[starting_index])

    df_euler = pd.DataFrame.from_dict(dict_euler)

    return df_euler
    

def main() :

    # some imports here, as they could be potentially used only in this main,
    # and not by the other functions
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Python script to create a dataset modified by the (Explicit) Euler Method, starting from a dataset created by an ODE system. By Alberto Tonda, 2022 <alberto.tonda@gmail.com>')
    parser.add_argument("--csv", metavar="dataset.csv", type=str, nargs=1, help="Dataset in CSV format. Must contain a column named \"t\", for time values.", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv[0])
    print(df)

    df_euler = apply_euler_method(df, delta_t=2)
    original_path, original_file = os.path.split(args.csv[0])
    df_euler_file = original_file[:-4] + "_euler.csv"

    print("Saving modified dataset to \"%s\"..." % df_euler_file)
    df_euler.to_csv(os.path.join(original_path, df_euler_file), index=False)

    return

if __name__ == "__main__" :
    sys.exit( main() )
