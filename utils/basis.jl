# this is the file containing the ChatGPT-generated code for the original 'basis.py' D-CODE script
module Basis

export FourierBasis, set_current_basis!, phi_t, dphi_dt, design_matrix, get_nonzero_range

using LinearAlgebra

abstract type BasisFunction end

mutable struct FourierBasis <: BasisFunction
    T::Float64
    n_basis::Int
    current_basis::Union{Int, Nothing}
    sqrt_2::Float64

    function FourierBasis(T::Float64, order::Int)
        new(T, order, nothing, sqrt(2))
    end
end

function set_current_basis!(basis::FourierBasis, i::Int)
    @assert i < basis.n_basis "Index out of bounds"
    basis.current_basis = i + 1
end

function phi_t(basis::FourierBasis, t::Float64)
    @assert basis.current_basis !== nothing "Current basis not set"
    t = t / basis.T
    return basis.sqrt_2 * sin(basis.current_basis * π * t)
end

function dphi_dt(basis::FourierBasis, t::Float64)
    @assert basis.current_basis !== nothing "Current basis not set"
    t = t / basis.T
    return basis.sqrt_2 * cos(basis.current_basis * π * t) * basis.current_basis * π / basis.T
end

function design_matrix(basis::FourierBasis, t::Vector{Float64}; derivative::Bool=false)
    save = basis.current_basis

    mat = [
        (set_current_basis!(basis, i); derivative ? dphi_dt(basis, ti) : phi_t(basis, ti))
        for ti in t, i in 0:(basis.n_basis - 1)
    ]

    basis.current_basis = save
    return mat
end

function get_nonzero_range(basis::FourierBasis)
    return (0.0, basis.T)
end

end  # End of module
