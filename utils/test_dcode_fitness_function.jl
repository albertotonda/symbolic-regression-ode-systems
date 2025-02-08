# let's try to reproduce the D-CODE fitness function

# imports
using JSON
using SymPy

# hard-coded values
odebench_file_name = "data/odebench/all_odebench_trajectories.json" # working directory is the main repo folder, for some reason

# first, we are going to read some data
println("Starting script! The working directory is:", pwd())
println("Loading file \"$odebench_file_name\"...")
odebench = JSON.parsefile(odebench_file_name)

# next, we iterate over the different systems in odebench
for system in odebench
    system_id = system["id"]
    println("Now working on system $system_id...")

    state_variables = [m.captures[1] for m in eachmatch(r"([0-9a-z_]+):\s+", system["var_description"])]
    println("State variables: $state_variables")

    equations = Dict(
        state_variables[i] => sympy.sympify(system["substituted"][1][i]) # array indexing begins at 1
        for i in eachindex(state_variables)
    )

    for (var, eq) in equations
        print("\t$var : $eq")
    end

    

    exit(0) # TODO remove this, for the moment it's just used for debugging
end
