module FlexPOI

import JuMP

const MOI = JuMP.MOI
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

export Optimizer
export diff_model
export nonlinear_diff_model
export conic_diff_model
export quadratic_diff_model

include("optimizer/core.jl")
include("optimizer/supports.jl")
include("optimizer/runtime.jl")

include("parameters.jl")

include("attributes/model.jl")
include("attributes/constraints.jl")

include("transform.jl")

include("copy.jl")

include("updates.jl")

include("cache/core.jl")
include("cache/constraint_updates.jl")

function _diffopt_extension_error(name::Symbol)
    error("`FlexPOI.$name` requires `using DiffOpt` to load the DiffOpt extension.")
end

function diff_model(args...; kwargs...)
    return _diffopt_extension_error(:diff_model)
end

function nonlinear_diff_model(args...; kwargs...)
    return _diffopt_extension_error(:nonlinear_diff_model)
end

function conic_diff_model(args...; kwargs...)
    return _diffopt_extension_error(:conic_diff_model)
end

function quadratic_diff_model(args...; kwargs...)
    return _diffopt_extension_error(:quadratic_diff_model)
end

end # module FlexPOI
