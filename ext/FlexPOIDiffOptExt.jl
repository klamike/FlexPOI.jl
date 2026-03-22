module FlexPOIDiffOptExt

import DiffOpt
import JuMP
import FlexPOI

const MOI = JuMP.MOI
const MOIU = MOI.Utilities

include("FlexPOIDiffOptExt/core.jl")
include("FlexPOIDiffOptExt/algebra.jl")
include("FlexPOIDiffOptExt/api.jl")
include("FlexPOIDiffOptExt/differentiate.jl")

function FlexPOI.diff_model(constructor)
    optimizer = FlexPOI.Optimizer(
        () -> DiffOpt.diff_optimizer(
            constructor;
            allow_parametric_opt_interface = false,
        ),
    )
    return JuMP.direct_model(optimizer)
end


end # module FlexPOIDiffOptExt
