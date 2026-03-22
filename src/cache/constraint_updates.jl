function _replace_scalar_constraint!(
    model::Optimizer{T},
    outer_ci::MOI.ConstraintIndex,
    cache::_ScalarConstraintCache{T},
    new_function,
    new_set,
) where {T}
    MOI.delete(model.optimizer, cache.inner_index)
    new_inner_index = _add_prepared_constraint(model, new_function, new_set)
    model.outer_to_inner_constraints[outer_ci] = new_inner_index
    _copy_constraint_attributes!(
        model.outer_model,
        model.optimizer,
        outer_ci,
        new_inner_index,
    )
    cache.inner_index = new_inner_index
    cache.current_function = new_function
    cache.current_set = new_set
    return
end

function _update_scalar_constraint_cache!(
    model::Optimizer{T},
    outer_ci::MOI.ConstraintIndex,
    cache::_ScalarConstraintCache{T},
    parameter_values::Dict{MOI.VariableIndex,T},
) where {T}
    label = _constraint_label(model.outer_model, outer_ci)
    new_function, new_set = _prepare_constraint_data(
        T,
        parameter_values,
        model.outer_to_inner_variables,
        cache.outer_function,
        cache.outer_set,
        label,
    )
    if typeof(new_function) != typeof(cache.current_function) ||
       typeof(new_set) != typeof(cache.current_set)
        _replace_scalar_constraint!(model, outer_ci, cache, new_function, new_set)
        return
    end
    if new_function isa MOI.ScalarNonlinearFunction
        if !isequal(cache.current_function, new_function) || !isequal(cache.current_set, new_set)
            _replace_scalar_constraint!(model, outer_ci, cache, new_function, new_set)
        end
        return
    end
    if !isequal(cache.current_set, new_set)
        MOI.set(model.optimizer, MOI.ConstraintSet(), cache.inner_index, new_set)
    end
    if new_function isa MOI.ScalarAffineFunction{T}
        _apply_scalar_affine_updates!(
            model.optimizer,
            cache.inner_index,
            cache.current_function,
            new_function,
        )
    else
        _apply_scalar_quadratic_updates!(
            model.optimizer,
            cache.inner_index,
            cache.current_function,
            new_function,
        )
    end
    cache.current_function = new_function
    cache.current_set = new_set
    return
end

function _replace_vector_constraint!(
    model::Optimizer,
    outer_ci::MOI.ConstraintIndex,
    cache::_VectorConstraintCache,
    new_function,
    new_set,
)
    MOI.delete(model.optimizer, cache.inner_index)
    new_inner_index = _add_prepared_constraint(model, new_function, new_set)
    model.outer_to_inner_constraints[outer_ci] = new_inner_index
    _copy_constraint_attributes!(
        model.outer_model,
        model.optimizer,
        outer_ci,
        new_inner_index,
    )
    cache.inner_index = new_inner_index
    cache.current_function = new_function
    cache.current_set = new_set
    return
end

function _update_vector_constraint_cache!(
    model::Optimizer{T},
    outer_ci::MOI.ConstraintIndex,
    cache::_VectorConstraintCache,
    parameter_values::Dict{MOI.VariableIndex,T},
) where {T}
    label = _constraint_label(model.outer_model, outer_ci)
    new_function, new_set = _prepare_constraint_data(
        T,
        parameter_values,
        model.outer_to_inner_variables,
        cache.outer_function,
        cache.outer_set,
        label,
    )
    if typeof(new_function) != typeof(cache.current_function) ||
       typeof(new_set) != typeof(cache.current_set)
        _replace_vector_constraint!(model, outer_ci, cache, new_function, new_set)
        return
    end
    if !isequal(cache.current_set, new_set)
        _replace_vector_constraint!(model, outer_ci, cache, new_function, new_set)
        return
    end
    if new_function isa MOI.VectorAffineFunction{T}
        _apply_vector_affine_updates!(
            model.optimizer,
            cache.inner_index,
            cache.current_function,
            new_function,
        )
    elseif new_function isa MOI.VectorQuadraticFunction{T}
        if _vector_quadratic_coefficients(cache.current_function) !=
           _vector_quadratic_coefficients(new_function)
            _replace_vector_constraint!(model, outer_ci, cache, new_function, new_set)
            return
        end
        _apply_vector_quadratic_updates!(
            model.optimizer,
            cache.inner_index,
            cache.current_function,
            new_function,
        )
    elseif !isequal(cache.current_function, new_function)
        _replace_vector_constraint!(model, outer_ci, cache, new_function, new_set)
        return
    else
        return
    end
    cache.current_function = new_function
    cache.current_set = new_set
    return
end
