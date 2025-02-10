# let's try to reproduce the D-CODE fitness function

# imports
using JSON
using SymPy

# these two below are imports from local scripts, originally in D-CODE
include("basis.jl")
using .Basis

# hard-coded values
odebench_file_name = "data/odebench/all_odebench_trajectories.json" # working directory is the main repo folder, for some reason
n_basis = 10
basis = Basis.FourierBasis

# first, we are going to read some data
println("Starting script! The working directory is:", pwd())
println("Loading file \"$odebench_file_name\"...")
odebench = JSON.parsefile(odebench_file_name)

# next, we iterate over the different systems in odebench
for system in odebench
    system_id = system["id"]
    println("Now working on system $system_id...")

    # detect state variables, get related equations in symbolic format
    state_variables = [m.captures[1] for m in eachmatch(r"([0-9a-z_]+):\s+", system["var_description"])]
    println("State variables: $state_variables")

    equations = Dict(
        state_variables[i] => sympify(system["substituted"][1][i]) # array indexing begins at 1
        for i in eachindex(state_variables)
    )

    for (var, eq) in equations
        println("\t$var : $eq")
    end

    # get the trajectories
    trajectories = system["solutions"][1]
    trajectories_array = nothing
    time_array = nothing

    for (trajectory_index, trajectory) in enumerate(trajectories)
        if trajectories_array === nothing
            # Prepare the array with the trajectories in a format that c_x expects
            # The shape is (n_samples, n_initial_conditions, n_variables)
            trajectories_array = zeros(
                length(trajectory["y"][1]),
                length(trajectories),
                length(state_variables)
            )
            # Also store the values of time, they will be used later
            time_array = zeros(length(trajectory["t"]), length(trajectories))
        end

        for (variable_index, variable) in enumerate(state_variables)
            trajectories_array[:, trajectory_index, variable_index] .= trajectory["y"][variable_index]  # Julia is 1-based
        end
        
        time_array[:, trajectory_index] .= trajectory["t"]
    end
    
    println("Found $(size(trajectories_array, 2)) trajectories, stored in a $(size(trajectories_array)) array")

    # here below, we proceed to the computation of c_x
    println("Computing c_x...")
    T = time_array[end, 1]
    t_new = time_array[:, 1]

    # get integration weights (created by ChatGPT)
    weight = ones(length(t_new))  # Equivalent to np.ones_like(t_new)
    weight[1] /= 2  # First element
    weight[end] /= 2  # Last element
    weight .= weight ./ sum(weight) .* T  # Normalize and scale by T

    # now, unfortunately we have to use the re-implementation of Python script originally in D-CODE
    # apparently, Julia likes structures and functions instead of classes
    basis_func = basis(T, n_basis)
    g_dot = Basis.design_matrix(basis_func, t_new, derivative=true)
    g = Basis.design_matrix(basis_func, t_new, derivative=false)
    
    # we should probably start iterating over variables here?
    # all the other parts are not dependant on the index of the variable
    for (variable_index, state_variable) in enumerate(state_variables)
        # compute c for all trajectories of one specific variable
        Xi = trajectories_array[:, :, variable_index]
        c = (Xi .* weight)' * g_dot
        
        # now, we can compute the value of the fitness function c_x for
        # the ground truth equation on all trajectories
        # this code is mostly cut/pasted from D-CODE
        x_hat = trajectories_array
        T, B, D = size(x_hat)
        x_hat_long = reshape(x_hat, (T * B, D))
        
        # compute values using the ground truth equation and sympy
        equation = equations[state_variable]
        symbols = [sympy.sympify(s) for s in state_variables]
        symbol_values = [x_hat_long[:,i] for i in 1:length(state_variables)]
        lambdified_function = lambdify(equation, symbols)
        #println(lambdified_function.(symbol_values...)) # the '.' should apply the function element-wise
        y_hat_long = lambdified_function.(symbol_values...)
        
        # finally obtain fitness function
        y_hat = reshape(y_hat_long, (T, B))
        c_hat = (y_hat .* weight)' * g
        
        # sample_weight was an array of ones, even in the original
        sample_weight = ones(size(c))
        c_x_fitness_value = sum((c + c_hat) .^ 2 .* sample_weight) / sum(sample_weight)
        
        println("$state_variable -> C_x fitness value for all trajectories: $(round(c_x_fitness_value, digits=10))")
    end

    exit(0) # TODO remove this, for the moment it's just used for debugging
end
