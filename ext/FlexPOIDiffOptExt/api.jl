function _parameter_error(prefix::AbstractString)
    return error(
        prefix * " when symbolic parameters are present. " *
        "Use `DiffOpt.ForwardConstraintSet()` / `DiffOpt.ReverseConstraintSet()` instead.",
    )
end

MOI.supports(::FlexPOI.Optimizer, ::DiffOpt.ForwardObjectiveFunction) = false
MOI.supports(::FlexPOI.Optimizer, ::DiffOpt.ReverseObjectiveFunction) = false

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardConstraintFunction,
    ::Type{<:MOI.ConstraintIndex},
)
    return false
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ReverseConstraintFunction,
    ::Type{<:MOI.ConstraintIndex},
)
    return false
end

function MOI.set(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardObjectiveFunction,
    _,
)
    return _parameter_error("Forward objective function is not supported")
end

function MOI.get(::FlexPOI.Optimizer, ::DiffOpt.ReverseObjectiveFunction)
    return _parameter_error("Reverse objective function is not supported")
end

function MOI.set(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardConstraintFunction,
    ::MOI.ConstraintIndex,
    _,
)
    return _parameter_error("Forward constraint function is not supported")
end

function MOI.get(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ReverseConstraintFunction,
    ::MOI.ConstraintIndex,
)
    return _parameter_error("Reverse constraint function is not supported")
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardConstraintSet,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}},
) where {T}
    return true
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ReverseConstraintSet,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}},
) where {T}
    return true
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardVariablePrimal,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ReverseVariablePrimal,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ForwardConstraintDual,
    ::Type{<:MOI.ConstraintIndex},
)
    return true
end

function MOI.supports(
    ::FlexPOI.Optimizer,
    ::DiffOpt.ReverseConstraintDual,
    ::Type{<:MOI.ConstraintIndex},
)
    return true
end

function DiffOpt.empty_input_sensitivities!(model::FlexPOI.Optimizer{T}) where {T}
    _require_diffopt_inner(model, DiffOpt.empty_input_sensitivities!)
    DiffOpt.empty_input_sensitivities!(model.optimizer)
    model.ext[_SENSITIVITY_DATA] = SensitivityData{T}()
    return
end

function MOI.set(
    model::FlexPOI.Optimizer{T},
    ::DiffOpt.ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    set::MOI.Parameter,
) where {T}
    variable = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
    sensitivity_data = _get_sensitivity_data(model)
    sensitivity_data.parameter_input_forward[variable] = convert(T, set.value)
    return
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    ::DiffOpt.ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    variable = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
    sensitivity_data = _get_sensitivity_data(model)
    return MOI.Parameter{T}(
        get(sensitivity_data.parameter_input_forward, variable, zero(T)),
    )
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    ::DiffOpt.ReverseConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    variable = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
    sensitivity_data = _get_sensitivity_data(model)
    return MOI.Parameter{T}(
        get(sensitivity_data.parameter_output_backward, variable, zero(T)),
    )
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ForwardVariablePrimal,
    variable::MOI.VariableIndex,
) where {T}
    if FlexPOI._is_parameter(model, variable)
        error("Trying to get a forward variable sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_variables[variable])
end

function MOI.set(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ReverseVariablePrimal,
    variable::MOI.VariableIndex,
    value,
) where {T}
    if FlexPOI._is_parameter(model, variable)
        error("Trying to set a backward variable sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    MOI.set(model.optimizer, attr, model.outer_to_inner_variables[variable], value)
    return
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ReverseVariablePrimal,
    variable::MOI.VariableIndex,
) where {T}
    if FlexPOI._is_parameter(model, variable)
        error("Trying to get a backward variable sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_variables[variable])
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ForwardConstraintDual,
    ci::MOI.ConstraintIndex,
) where {T}
    if _is_parameter_constraint(T, ci)
        error("Trying to get a forward constraint dual sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
end

function MOI.set(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ReverseConstraintDual,
    ci::MOI.ConstraintIndex,
    value,
) where {T}
    if _is_parameter_constraint(T, ci)
        error("Trying to set a backward constraint dual sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    MOI.set(model.optimizer, attr, model.outer_to_inner_constraints[ci], value)
    return
end

function MOI.get(
    model::FlexPOI.Optimizer{T},
    attr::DiffOpt.ReverseConstraintDual,
    ci::MOI.ConstraintIndex,
) where {T}
    if _is_parameter_constraint(T, ci)
        error("Trying to get a backward constraint dual sensitivity for a parameter")
    end
    FlexPOI._require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
end

function MOI.set(
    model::FlexPOI.Optimizer,
    attr::DiffOpt.ReverseObjectiveSensitivity,
    value,
)
    _require_diffopt_inner(model, DiffOpt.reverse_differentiate!)
    MOI.set(model.optimizer, attr, value)
    return
end

function MOI.get(
    model::FlexPOI.Optimizer,
    ::DiffOpt.ForwardObjectiveSensitivity,
)
    _require_diffopt_inner(model, DiffOpt.forward_differentiate!)
    return _forward_objective_sensitivity_fallback(model)
end

function MOI.get(
    model::FlexPOI.Optimizer,
    attr::DiffOpt.DifferentiateTimeSec,
)
    _require_diffopt_inner(model, DiffOpt.forward_differentiate!)
    return MOI.get(model.optimizer, attr)
end
