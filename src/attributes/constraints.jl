function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
)
    if MOI.is_set_by_optimize(attr)
        if ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}
            throw(MOI.GetAttributeNotAllowed(attr, "No result is available for parameter constraints."))
        end
        _require_result(model, attr)
        return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
    end
    return MOI.get(model.outer_model, attr, ci)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    if MOI.is_set_by_optimize(attr)
        if ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}
            throw(MOI.SetAttributeNotAllowed(attr, "Cannot set result attributes on parameter constraints."))
        end
        MOI.set(model.optimizer, attr, model.outer_to_inner_constraints[ci], value)
        return
    end
    if attr isa Union{MOI.ConstraintSet,MOI.ConstraintFunction}
        MOI.set(model.outer_model, attr, ci, value)
        _invalidate_solution!(model)
        if ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter} || model.structure_dirty
            return
        end
        try
            _sync_constraint_cache!(model, ci; parameter_values = model.parameter_values)
        catch err
            _should_retry_full_rebuild(err) || rethrow()
            _invalidate_structure!(model)
        end
        return
    end
    MOI.set(model.outer_model, attr, ci, value)
    if !(ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}) &&
       !model.structure_dirty &&
       haskey(model.outer_to_inner_constraints, ci) &&
       !MOI.is_set_by_optimize(attr) &&
       MOI.supports(model.optimizer, attr, typeof(model.outer_to_inner_constraints[ci]))
        try
            MOI.set(model.optimizer, attr, model.outer_to_inner_constraints[ci], value)
        catch err
            err isa MOI.UnsupportedError && return
            err isa MOI.NotAllowedError && return
            rethrow()
        end
    end
    return
end

function MOI.set(
    model::Optimizer{T},
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    value::MOI.Parameter{T},
) where {T}
    MOI.set(model.outer_model, MOI.ConstraintSet(), ci, value)
    vi = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
    model.pending_parameter_values[vi] = value.value
    _invalidate_solution!(model)
    return
end

function MOI.get(model::Optimizer{T}, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex) where {T}
    if ci isa MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}
        vi = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
        return _parameter_value(model, vi)
    end
    _require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    if ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}
        throw(MOI.GetAttributeNotAllowed(attr, "Duals are not defined for parameter constraints."))
    end
    _require_result(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintConflictStatus,
    ci::MOI.ConstraintIndex,
)
    if ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}
        return MOI.NOT_IN_CONFLICT
    end
    _require_conflict(model, attr)
    return MOI.get(model.optimizer, attr, model.outer_to_inner_constraints[ci])
end
