mutable struct SensitivityData{T}
    parameter_input_forward::Dict{MOI.VariableIndex,T}
    parameter_output_backward::Dict{MOI.VariableIndex,T}
end

function SensitivityData{T}() where {T}
    return SensitivityData{T}(
        Dict{MOI.VariableIndex,T}(),
        Dict{MOI.VariableIndex,T}(),
    )
end

const _SENSITIVITY_DATA = :_diffopt_sensitivity_data

function _get_sensitivity_data(
    model::FlexPOI.Optimizer{T},
)::SensitivityData{T} where {T}
    if !haskey(model.ext, _SENSITIVITY_DATA)
        model.ext[_SENSITIVITY_DATA] = SensitivityData{T}()
    end
    return model.ext[_SENSITIVITY_DATA]::SensitivityData{T}
end

function _require_diffopt_inner(model::FlexPOI.Optimizer, action::Function)
    if applicable(action, model.optimizer)
        return
    end
    error(
        "FlexPOI DiffOpt integration requires a DiffOpt-capable inner optimizer. " *
        "Construct the model with `FlexPOI.Optimizer(() -> DiffOpt.diff_optimizer(solver; " *
        "allow_parametric_opt_interface = false))`.",
    )
end

function _parameter_variables(model::FlexPOI.Optimizer{T}) where {T}
    cis = MOI.get(
        model.outer_model,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Parameter{T}}(),
    )
    return [MOI.get(model.outer_model, MOI.ConstraintFunction(), ci) for ci in cis]
end

_is_parameter_constraint(::Type{T}, ci::MOI.ConstraintIndex) where {T} =
    ci isa MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}

function _is_zero_function(value::Bool)
    return !value
end

function _is_zero_function(value::Real)
    return iszero(value)
end

function _is_zero_function(func::MOI.ScalarAffineFunction)
    return isempty(func.terms) && iszero(MOI.constant(func))
end

function _is_zero_function(func::MOI.ScalarQuadraticFunction)
    return isempty(func.quadratic_terms) &&
           isempty(func.affine_terms) &&
           iszero(MOI.constant(func))
end

function _is_zero_function(func::MOI.VectorAffineFunction)
    return isempty(func.terms) && all(iszero, func.constants)
end

function _is_zero_function(func::MOI.VectorQuadraticFunction)
    return isempty(func.quadratic_terms) &&
           isempty(func.affine_terms) &&
           all(iszero, func.constants)
end

function _simplify_derivative(raw)
    if raw isa Union{
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    }
        return MOI.Nonlinear.SymbolicAD.simplify(raw)
    end
    return raw
end

function _transform_scalar_derivative(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    raw,
    context::AbstractString,
) where {T}
    if raw === false
        return nothing
    elseif raw === true
        return FlexPOI._scalar_constant(T, one(T))
    elseif raw isa Real
        return FlexPOI._scalar_constant(T, raw)
    elseif raw isa Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    }
        transformed = FlexPOI._transform_function(
            T,
            parameter_values,
            variable_map,
            raw,
        )
        if transformed isa MOI.ScalarNonlinearFunction
            error(
                "DiffOpt parameter perturbations for $context must simplify to an affine " *
                "or quadratic scalar function. Simplification produced: " *
                string(transformed),
            )
        end
        normalized = FlexPOI._normalize_scalar_result(T, transformed)
        return _is_zero_function(normalized) ? nothing : normalized
    end
    error(
        "Unsupported DiffOpt parameter derivative for $context: " *
        string(typeof(raw)),
    )
end

function _scalar_partial_derivative(
    model::FlexPOI.Optimizer{T},
    func,
    parameter::MOI.VariableIndex,
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    context::AbstractString,
) where {T}
    raw = _simplify_derivative(MOI.Nonlinear.SymbolicAD.derivative(func, parameter))
    return _transform_scalar_derivative(
        T,
        parameter_values,
        variable_map,
        raw,
        context,
    )
end

function _vector_partial_derivative(
    model::FlexPOI.Optimizer{T},
    func::MOI.AbstractVectorFunction,
    parameter::MOI.VariableIndex,
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    context::AbstractString,
) where {T}
    rows = MOIU.scalarize(func)
    perturbation_rows = MOI.ScalarAffineFunction{T}[]
    sizehint!(perturbation_rows, length(rows))
    for (i, row) in enumerate(rows)
        row_context = "$context row $i"
        partial = _scalar_partial_derivative(
            model,
            row,
            parameter,
            parameter_values,
            variable_map,
            row_context,
        )
        if partial === nothing
            push!(perturbation_rows, FlexPOI._scalar_constant(T, zero(T)))
        elseif partial isa MOI.ScalarAffineFunction{T}
            push!(perturbation_rows, partial)
        elseif partial isa MOI.ScalarAffineFunction
            push!(perturbation_rows, convert(MOI.ScalarAffineFunction{T}, partial))
        else
            error(
                "DiffOpt parameter perturbations for vector constraints must be affine. " *
                "$row_context simplified to $(typeof(partial)).",
            )
        end
    end
    perturbation = MOIU.vectorize(perturbation_rows)
    return _is_zero_function(perturbation) ? nothing : perturbation
end

function _parameter_partial_derivative(
    model::FlexPOI.Optimizer{T},
    func::MOI.AbstractScalarFunction,
    parameter::MOI.VariableIndex,
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    context::AbstractString,
) where {T}
    return _scalar_partial_derivative(
        model,
        func,
        parameter,
        parameter_values,
        variable_map,
        context,
    )
end

function _parameter_partial_derivative(
    model::FlexPOI.Optimizer{T},
    func::MOI.AbstractVectorFunction,
    parameter::MOI.VariableIndex,
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    context::AbstractString,
) where {T}
    return _vector_partial_derivative(
        model,
        func,
        parameter,
        parameter_values,
        variable_map,
        context,
    )
end

function _add_scaled_perturbation(::Type{T}, total, piece, α) where {T}
    scaled = isone(α) ? piece : MOIU.operate(*, T, convert(T, α), piece)
    if total === nothing
        return scaled
    end
    return MOIU.operate(+, T, total, scaled)
end

function _build_forward_perturbation(
    model::FlexPOI.Optimizer{T},
    func::MOI.AbstractFunction,
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    context::AbstractString,
) where {T}
    sensitivity_data = _get_sensitivity_data(model)
    total = nothing
    for (parameter, direction) in sensitivity_data.parameter_input_forward
        iszero(direction) && continue
        partial = _parameter_partial_derivative(
            model,
            func,
            parameter,
            parameter_values,
            variable_map,
            context,
        )
        partial === nothing && continue
        total = _add_scaled_perturbation(T, total, partial, direction)
    end
    return total
end
