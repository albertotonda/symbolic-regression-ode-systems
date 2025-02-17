# full test of SymbolicRegression.jl with the new fitness function and the new options

# imports
using Base.Filesystem # create directories and such
using Dates # manage dates
using JSON
using Logging # for logging
using SymPy

import SymbolicRegression: Dataset, Options, calculate_pareto_frontier, equation_search, tree, eval_tree_array
import SymbolicRegression

# these two below are imports from local scripts, originally in D-CODE
include("basis.jl")
using .Basis

# hard-coded values
odebench_file_name = "data/odebench/all_odebench_trajectories.json" # working directory is the main repo folder, for some reason
output_folder = "local_results/julia-experiments"
n_basis = 50
basis = Basis.FourierBasis
niterations = 40

# first, let's create the output directory
timestamp = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
output_folder = joinpath(output_folder, timestamp * "-julia-experiments") # string concatenation in Julia uses '*'
mkpath(output_folder)

# then, start the log file
logger = SimpleLogger(open(joinpath(output_folder, "log.txt"), "w"))
global_logger(logger)

@info("Preparing structures and functions...")
# all this block creates the MyNewOptions structure, and a merged structure MyOptions with the previous SymbolicRegression.Options
Base.@kwdef struct MyNewOptions
    c::Array{Float64} = nothing
    g::Array{Float64} = nothing
    x_hat::Array{Float64} = nothing
    weight::Array{Float64} = nothing
end

struct MyOptions{O<:SymbolicRegression.Options} <: SymbolicRegression.AbstractOptions
    new_options::MyNewOptions
    sr_options::O
end
const NEW_OPTIONS_KEYS = fieldnames(MyNewOptions)

# Constructor with both sets of parameters:
function MyOptions(; kws...)
    new_options_keys = filter(k -> k in NEW_OPTIONS_KEYS, keys(kws))
    new_options = MyNewOptions(; NamedTuple(new_options_keys .=> Tuple(kws[k] for k in new_options_keys))...)
    sr_options_keys = filter(k -> !(k in NEW_OPTIONS_KEYS), keys(kws))
    sr_options = SymbolicRegression.Options(; NamedTuple(sr_options_keys .=> Tuple(kws[k] for k in sr_options_keys))...)
    return MyOptions(new_options, sr_options)
end

# Make all `Options` available while also making `new_options` accessible
function Base.getproperty(options::MyOptions, k::Symbol)
    if k in NEW_OPTIONS_KEYS
        return getproperty(getfield(options, :new_options), k)
    else
        return getproperty(getfield(options, :sr_options), k)
    end
end

Base.propertynames(options::MyOptions) = (NEW_OPTIONS_KEYS..., fieldnames(SymbolicRegression.Options)...)
# end of the big block defining the new Options

# and now a custom loss function, based on the D-CODE paper
#function dcode_loss(model::SymbolicRegression.SRRegressor, X, y, options::MyOptions)
function dcode_loss(tree, dataset::Dataset{T,L}, options::MyOptions)::L where {T,L}
    # here we are going to use the extra information from options.new_options,
    # that should contain the precomputed values of c and g taken from the trajectories
    c = options.c
    g = options.g
    x_hat = options.x_hat
    weight = options.weight

    # obtain all the necessary values and compute c_hat!
    # the first step is to obtain y_hat, which is the predicted value for x_hat
    T1, B, D = size(x_hat)
    x_hat_long = reshape(x_hat, (T1 * B, D))
    
    # we need to use the try/catch block because the evaluation of the tree can raise exceptions
    flag = true
    y_hat_long = nothing
    try
        # x_hat_long needs to be transposed, because eval_tree_array expects a shape (n_features, n_samples)
        y_hat_long, flag = eval_tree_array(tree, x_hat_long', options)
    catch e
        @error("Unhandled exception during tree evaluation: \"$e\"")
        flag = false
    end

    # flag == false means that something went wrong during the evaluation of the tree,
    # so very likely the fitness is NaN or infinite
    if flag == false
        return L(Inf)
    end
    
    y_hat = reshape(y_hat_long, (T1, B))
    c_hat = (y_hat .* weight)' * g

    # compute final value of the fitness function
    sample_weight = ones(size(c))
    c_x_fitness_value = sum((c + c_hat) .^ 2 .* sample_weight) / sum(sample_weight)

    return L(c_x_fitness_value)
end

# first, we are going to read some data
@info("Starting script! The working directory is:", pwd())
@info("Loading file \"$odebench_file_name\"...")
odebench = JSON.parsefile(odebench_file_name)

# next, we iterate over the different systems in odebench
for system in odebench
    system_id = system["id"]
    @info("Now working on system $system_id...")

    # detect state variables, get related equations in symbolic format
    state_variables = [m.captures[1] for m in eachmatch(r"([0-9a-z_]+):\s+", system["var_description"])]
    @info("State variables: $state_variables")

    equations = Dict(
        state_variables[i] => sympify(system["substituted"][1][i]) # array indexing begins at 1
        for i in eachindex(state_variables)
    )

    for (var, eq) in equations
        @info("\t$var : $eq")
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
    
    @info("Found $(size(trajectories_array, 2)) trajectories, stored in a $(size(trajectories_array)) array")

    # here below, we proceed to the computation of c_x
    @info("Preparing the computation of c_x...")
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

    # iterate over the variables, run symbolic regression
    for (variable_index, state_variable) in enumerate(state_variables)
        # compute c for all trajectories of one specific variable
        Xi = trajectories_array[:, :, variable_index]
        c = (Xi .* weight)' * g_dot

        # create the output directory for this system and variable
        variable_folder = joinpath(output_folder, "system-$system_id-$state_variable")
        mkpath(variable_folder)

        # set up the options for SymbolicRegression.jl
        options = MyOptions(
            c = c,
            g = g,
            x_hat = trajectories_array,
            weight = weight,
            # these are the basic options for SymbolicRegression.jl
            binary_operators=[+, -, *, /],
            unary_operators=[sin, cos, exp, log],
            populations=20,
            loss_function = dcode_loss,
            output_directory = variable_folder
        )
        #println("Trying to access options field: $(options.c)")
        
        # after mocking the choices of the D-CODE scripts, we actually fall into 
        # the same pattern, and create two useless empty structures; so that the
        # checks performed by SymbolicRegression on the shapes are satisfied (hopefully)
        X = zeros(3, 150)
        y = zeros(150)

        # run the search
        @info("Running Symbolic Regression...")
        hall_of_fame = equation_search(
            X, y, 
            niterations=niterations,
            variable_names=String.(state_variables), # this is necessary to convert SubString into String
            y_variable_names=String.(state_variable), # again, same as above
            options=options,
        )

        @info("Final hall of fame: $hall_of_fame")
    end

    # TODO comment/remove this, it's just for debugging
    #if system_id == 2
    #    exit(0)
    #end
end