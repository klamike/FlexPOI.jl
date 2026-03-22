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

function _diffopt_outer_model(
    constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
    allow_parametric_opt_interface = false,
)
    optimizer = FlexPOI.Optimizer(
        () -> DiffOpt.diff_optimizer(
            constructor;
            with_bridge_type,
            with_cache_type,
            with_outer_cache,
            allow_parametric_opt_interface,
        ),
    )
    return JuMP.direct_model(optimizer)
end

function _set_model_constructor!(model::JuMP.Model, constructor)
    MOI.set(JuMP.backend(model), DiffOpt.ModelConstructor(), constructor)
    return model
end

function FlexPOI.diff_model(
    constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    return _diffopt_outer_model(
        constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
end

function FlexPOI.nonlinear_diff_model(
    constructor;
    with_parametric_opt_interface = false,
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    model = _diffopt_outer_model(
        constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
        allow_parametric_opt_interface = with_parametric_opt_interface,
    )
    return _set_model_constructor!(model, DiffOpt.NonLinearProgram.Model)
end

function FlexPOI.conic_diff_model(
    constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    model = _diffopt_outer_model(
        constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    return _set_model_constructor!(model, DiffOpt.ConicProgram.Model)
end

function FlexPOI.quadratic_diff_model(
    constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    model = _diffopt_outer_model(
        constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    return _set_model_constructor!(model, DiffOpt.QuadraticProgram.Model)
end

end # module FlexPOIDiffOptExt
