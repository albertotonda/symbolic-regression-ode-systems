import re
import sympy
import sys

def parse_ode_from_json(file_name) :

    return

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

    return equations, initial_conditions, time_step, max_time

def main() :

    # these packages are only used in the main, so they are imported here, just
    # in case we will later import the script for its functions
    import argparse
    import os

    # parse arguments
    parser = argparse.ArgumentParser(description='Python script to create dataset files, starting from a text description of a dynamic system + initial conditions. By Alberto Tonda, 2022 <alberto.tonda@inrae.fr>')
    parser.add_argument("--input", metavar="system.txt", type=str, nargs=1, help="input file; format:\ndX_1/dt = <equation>.\ndX_2/dt = <quation>\n...\ndX_n/dt = <equation>\ninitial_conditions: X_1=x1 X_2=x2 ... X_n=xN\ntime_step: 0.1 <optional>i\nmax_time: 100", required=True)
    args = parser.parse_args()

    # get the new file name
    path, ode_file_name = os.path.split(args.input[0])
    dataset_file_name = ode_file_name[:-4] + "-dataset.csv"

    print("I am going to read file \"%s\" and write to file \"%s\"..." % (ode_file_name, dataset_file_name))

    # check if the file ends with ".txt" or ".json"
    if args.input[0].endswith(".txt") :
        equations, initial_conditions, time_step, max_time = parse_ode_from_text(args.input[0])

    elif args.input[0].endswith(".json") :
        parse_ode_from_json(args.input[0])

    else :
        print("The file for the ODE system should be in either .txt or .json format. Aborting...")
        sys.exit(0)


    # debug lines
    print("Equations:", equations)
    print("Initial conditions:", initial_conditions)
    print("time_step=%.4f, max_time=%.4f" % (time_step, max_time))


    return

if __name__ == "__main__" :
    sys.exit( main() )
