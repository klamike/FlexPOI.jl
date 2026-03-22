function _user_defined_operators(model::Optimizer)
    operators = Symbol[]
    for attr in MOI.get(model.outer_model, MOI.ListOfModelAttributesSet())
        if attr isa MOI.UserDefinedFunction
            push!(operators, attr.name)
        end
    end
    return operators
end

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    operators = copy(_SUPPORTED_NONLINEAR_OPERATORS)
    append!(operators, _user_defined_operators(model))
    unique!(operators)
    sort!(operators; by = string)
    return operators
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    return MOI.get(model.outer_model, MOI.VariableIndex, name)
end

function MOI.get(model::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    return MOI.get(model.outer_model, MOI.ConstraintIndex, name)
end

function MOI.get(
    model::Optimizer,
    ::Type{MOI.ConstraintIndex{F,S}},
    name::String,
) where {F,S}
    return MOI.get(model.outer_model, MOI.ConstraintIndex{F,S}, name)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfOptimizerAttributesSet)
    return collect(keys(model.optimizer_attributes))
end

function MOI.get(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    if haskey(model.optimizer_attributes, attr)
        return model.optimizer_attributes[attr]
    end
    return MOI.get(model.optimizer, attr)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    if !MOI.supports(model.optimizer, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    model.optimizer_attributes[attr] = value
    MOI.set(model.optimizer, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    if MOI.is_set_by_optimize(attr)
        _require_result(model, attr)
        return MOI.get(model.optimizer, attr)
    end
    return MOI.get(model.outer_model, attr)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractModelAttribute, value)
    if MOI.is_set_by_optimize(attr)
        MOI.set(model.optimizer, attr, value)
        return
    end
    if attr isa Union{MOI.ObjectiveSense,MOI.ObjectiveFunction}
        MOI.set(model.outer_model, attr, value)
        _invalidate_solution!(model)
        if model.structure_dirty
            return
        end
        try
            _sync_objective_cache!(
                model;
                parameter_values = model.parameter_values,
                force = true,
            )
        catch err
            _should_retry_full_rebuild(err) || rethrow()
            _invalidate_structure!(model)
        end
        return
    end
    MOI.set(model.outer_model, attr, value)
    if !model.structure_dirty && MOI.supports(model.optimizer, attr)
        index_map = _index_map(
            model.outer_to_inner_variables,
            model.outer_to_inner_constraints,
        )
        try
            MOI.set(model.optimizer, attr, MOIU.map_indices(index_map, attr, value))
        catch err
            err isa MOI.UnsupportedAttribute && return
            err isa MOI.UnsupportedError && return
            err isa MOI.NotAllowedError && return
            rethrow()
        end
    end
    return
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.result_available ? MOI.get(model.optimizer, MOI.ResultCount()) : 0
end

function MOI.get(model::Optimizer, ::MOI.ConflictStatus)
    return model.conflict_available ? MOI.get(model.optimizer, MOI.ConflictStatus()) :
           MOI.COMPUTE_CONFLICT_NOT_CALLED
end

function MOI.get(model::Optimizer, ::MOI.ConflictCount)
    return model.conflict_available ? MOI.get(model.optimizer, MOI.ConflictCount()) : 0
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return model.result_available ? MOI.get(model.optimizer, MOI.TerminationStatus()) :
           MOI.OPTIMIZE_NOT_CALLED
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    return model.result_available ? MOI.get(model.optimizer, attr) : MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    return model.result_available ? MOI.get(model.optimizer, attr) : MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::Union{
    MOI.RawStatusString,
    MOI.ObjectiveValue,
    MOI.ObjectiveBound,
    MOI.RelativeGap,
    MOI.SolveTimeSec,
    MOI.SimplexIterations,
    MOI.BarrierIterations,
    MOI.NodeCount,
})
    _require_result(model, attr)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex)
    if _is_parameter(model, vi)
        if attr isa MOI.VariablePrimal
            return _parameter_value(model, vi)
        elseif MOI.is_set_by_optimize(attr)
            throw(MOI.GetAttributeNotAllowed(attr, "No result is available for parameter variables."))
        end
        return MOI.get(model.outer_model, attr, vi)
    end
    if MOI.is_set_by_optimize(attr)
        _require_result(model, attr)
        return MOI.get(model.optimizer, attr, model.outer_to_inner_variables[vi])
    end
    return MOI.get(model.outer_model, attr, vi)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    if MOI.is_set_by_optimize(attr)
        if _is_parameter(model, vi)
            throw(MOI.SetAttributeNotAllowed(attr, "Cannot set result attributes on parameter variables."))
        end
        MOI.set(model.optimizer, attr, model.outer_to_inner_variables[vi], value)
        return
    end
    MOI.set(model.outer_model, attr, vi, value)
    if !_is_parameter(model, vi) &&
       !model.structure_dirty &&
       haskey(model.outer_to_inner_variables, vi) &&
       !MOI.is_set_by_optimize(attr) &&
       MOI.supports(model.optimizer, attr, MOI.VariableIndex)
        try
            MOI.set(model.optimizer, attr, model.outer_to_inner_variables[vi], value)
        catch err
            err isa MOI.UnsupportedError && return
            err isa MOI.NotAllowedError && return
            rethrow()
        end
    end
    return
end

function MOI.get(model::Optimizer{T}, attr::MOI.VariablePrimal, vi::MOI.VariableIndex) where {T}
    if _is_parameter(model, vi)
        return _parameter_value(model, vi)
    end
    _require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_variables[vi])
end
