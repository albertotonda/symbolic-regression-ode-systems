# this is an example using MLJ, some sort of common Machine Learning interface for Julia

# import SymbolicRegression: SRRegressor
# import MLJ: machine, fit!, predict, report

# # Dataset with two named features:
# X = (a = rand(500), b = rand(500))

# # and one target:
# y = @. 2 * cos(X.a * 23.5) - X.b ^ 2

# # with some noise:
# y = y .+ randn(500) .* 1e-3

# model = SRRegressor(
#     niterations=50,
#     binary_operators=[+, -, *],
#     unary_operators=[cos],
# )

# mach = machine(model, X, y)

# fit!(mach)
# report(mach)

# this below is another example, using the low-level API
# import SymbolicRegression: Options, calculate_pareto_frontier, equation_search

# X = randn(2, 100)
# y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

# options = Options(
#     binary_operators=[+, *, /, -],
#     unary_operators=[cos, exp],
#     populations=20
# )

# hall_of_fame = equation_search(
#     X, y, niterations=40, options=options,
#     parallelism=:multithreading
# )

# dominating = calculate_pareto_frontier(hall_of_fame)
# print(dominating)

# ok, I think that now the general idea is to:
# 1. instantiate a custom inherited class from AbstractOptions, containing the precomputed values of c() and g()
# 2. create a custom loss function which is reading the extra information from the AbstractOptions
# 3. perform all the necessary computations for c_hat inside the custom loss function

import SymbolicRegression: Options, calculate_pareto_frontier, equation_search
import SymbolicRegression
# all this block creates the MyNewOptions structure, and a merged structure MyOptions with the previous SymbolicRegression.Options
Base.@kwdef struct MyNewOptions
    a::Float64 = 1.0
    b::Int = 3
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

# and now a custom loss function
function custom_loss(model::SymbolicRegression.SRRegressor, X, y, options::MyOptions)
    # here we can use the extra information from options.new_options
    println(options.new_options.a)
    println(options.new_options.b)
    return SymbolicRegression.loss(model, X, y, options.sr_options)
end

# and now we test the whole thing; set up the options
options = MyOptions(
    a = 2.0,
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=20
)

# target and data
X = randn(2, 100)
y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

# run the search
hall_of_fame = equation_search(
    X, y, niterations=40, options=options,
    parallelism=:multithreading
)