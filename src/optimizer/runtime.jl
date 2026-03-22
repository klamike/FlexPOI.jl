function MOI.add_variable(model::Optimizer)
    _invalidate_structure!(model)
    return MOI.add_variable(model.outer_model)
end

function MOI.add_variables(model::Optimizer, n::Integer)
    _invalidate_structure!(model)
    return MOI.add_variables(model.outer_model, n)
end

function MOI.add_constrained_variable(model::Optimizer, set::MOI.AbstractScalarSet)
    _invalidate_structure!(model)
    return MOI.add_constrained_variable(model.outer_model, set)
end

function MOI.add_constrained_variable(model::Optimizer{T}, set::MOI.Parameter{T}) where {T}
    _invalidate_structure!(model)
    vi, ci = MOI.add_constrained_variable(model.outer_model, set)
    model.parameter_values[vi] = set.value
    model.parameter_constraints[vi] = ci
    return vi, ci
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::Tuple{MOI.GreaterThan,MOI.LessThan},
)
    _invalidate_structure!(model)
    return MOI.add_constrained_variable(model.outer_model, set)
end

function MOI.add_constrained_variables(
    model::Optimizer,
    sets::AbstractVector{<:MOI.AbstractScalarSet},
)
    _invalidate_structure!(model)
    result = MOI.add_constrained_variables(model.outer_model, sets)
    _sync_parameter_state!(model)
    return result
end

function MOI.add_constrained_variables(model::Optimizer, set::MOI.AbstractVectorSet)
    _invalidate_structure!(model)
    result = MOI.add_constrained_variables(model.outer_model, set)
    _sync_parameter_state!(model)
    return result
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    _invalidate_structure!(model)
    return MOI.add_constraint(model.outer_model, func, set)
end

function MOI.add_constraint(
    model::Optimizer,
    func::Vector{MOI.VariableIndex},
    set::MOI.AbstractVectorSet,
)
    _invalidate_structure!(model)
    if any(_is_parameter(model, vi) for vi in func)
        error("Parameter variables cannot participate in VectorOfVariables constraints.")
    end
    return MOI.add_constraint(model.outer_model, func, set)
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.VectorOfVariables,
    set::MOI.AbstractVectorSet,
)
    _invalidate_structure!(model)
    if any(_is_parameter(model, vi) for vi in func.variables)
        error("Parameter variables cannot participate in VectorOfVariables constraints.")
    end
    return MOI.add_constraint(model.outer_model, func, set)
end

function MOI.delete(
    model::Optimizer,
    index::Union{MOI.VariableIndex,MOI.ConstraintIndex},
)
    _invalidate_structure!(model)
    result = MOI.delete(model.outer_model, index)
    _sync_parameter_state!(model)
    return result
end

function MOI.modify(model::Optimizer, ci::MOI.ConstraintIndex, change::MOI.AbstractFunctionModification)
    MOI.modify(model.outer_model, ci, change)
    _invalidate_solution!(model)
    if model.structure_dirty || !haskey(model.outer_to_inner_constraints, ci)
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

function MOI.modify(
    model::Optimizer,
    attr::MOI.ObjectiveFunction,
    change::MOI.AbstractFunctionModification,
)
    try
        MOI.modify(model.outer_model, attr, change)
    catch err
        err isa MOI.ModifyObjectiveNotAllowed || rethrow()
        _modify_outer_objective!(model, attr, change)
    end
    _invalidate_solution!(model)
    if model.structure_dirty
        return
    end
    try
        _sync_objective_cache!(model; parameter_values = model.parameter_values, force = true)
    catch err
        _should_retry_full_rebuild(err) || rethrow()
        _invalidate_structure!(model)
    end
    return
end

function _build_inner_model!(model::Optimizer{T}) where {T}
    _reset_optimizer!(model)
    _clear_incremental_caches!(model)
    parameter_values = _parameter_values(model)
    variable_map = _copy_variables!(model, parameter_values)
    constraint_map = Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}()
    _copy_variable_constraints!(model, parameter_values, variable_map, constraint_map)
    _copy_model_attributes!(model, variable_map, constraint_map)
    _build_objective_cache!(model, parameter_values, variable_map)
    _build_function_constraint_caches!(
        model,
        parameter_values,
        variable_map,
        constraint_map,
    )
    model.outer_to_inner_variables = variable_map
    model.outer_to_inner_constraints = constraint_map
    _commit_parameter_values!(model, parameter_values)
    model.structure_dirty = false
    return
end

function _update_parameter_caches!(model::Optimizer{T}) where {T}
    parameter_values = _parameter_values(model)
    if _same_parameter_values(parameter_values, model.parameter_values)
        return
    end
    changed_parameters = _changed_parameter_variables(parameter_values, model.parameter_values)
    try
        _update_objective_cache!(
            model,
            parameter_values;
            force = false,
            changed_parameters,
        )
        for (outer_ci, cache) in model.scalar_constraint_caches
            cache.uses_parameters || continue
            _depends_on_changed_parameter(cache.parameter_dependencies, changed_parameters) ||
                continue
            _update_scalar_constraint_cache!(model, outer_ci, cache, parameter_values)
        end
        for (outer_ci, cache) in model.vector_constraint_caches
            cache.uses_parameters || continue
            _depends_on_changed_parameter(cache.parameter_dependencies, changed_parameters) ||
                continue
            _update_vector_constraint_cache!(model, outer_ci, cache, parameter_values)
        end
    catch err
        _should_retry_full_rebuild(err) || rethrow()
        _build_inner_model!(model)
        return
    end
    _commit_parameter_values!(model, parameter_values)
    return
end

function MOI.optimize!(model::Optimizer{T}) where {T}
    if model.structure_dirty
        _build_inner_model!(model)
    else
        _update_parameter_caches!(model)
    end
    model.conflict_available = false
    MOI.optimize!(model.optimizer)
    model.result_available = true
    return
end

function MOI.compute_conflict!(model::Optimizer)
    if model.structure_dirty
        _build_inner_model!(model)
    else
        _update_parameter_caches!(model)
    end
    MOI.compute_conflict!(model.optimizer)
    model.conflict_available = true
    return
end
