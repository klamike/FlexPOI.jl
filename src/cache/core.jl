function _sync_constraint_cache!(
    model::Optimizer{T},
    outer_ci::MOI.ConstraintIndex;
    parameter_values::Dict{MOI.VariableIndex,T},
) where {T}
    if haskey(model.scalar_constraint_caches, outer_ci)
        cache = model.scalar_constraint_caches[outer_ci]
        cache.outer_function = MOI.get(model.outer_model, MOI.ConstraintFunction(), outer_ci)
        cache.outer_set = MOI.get(model.outer_model, MOI.ConstraintSet(), outer_ci)
        cache.parameter_dependencies = _function_parameter_dependencies(
            parameter_values,
            cache.outer_function,
        )
        cache.uses_parameters = !isempty(cache.parameter_dependencies)
        _update_scalar_constraint_cache!(model, outer_ci, cache, parameter_values)
        return
    elseif haskey(model.vector_constraint_caches, outer_ci)
        cache = model.vector_constraint_caches[outer_ci]
        cache.outer_function = MOI.get(model.outer_model, MOI.ConstraintFunction(), outer_ci)
        cache.outer_set = MOI.get(model.outer_model, MOI.ConstraintSet(), outer_ci)
        cache.parameter_dependencies = _function_parameter_dependencies(
            parameter_values,
            cache.outer_function,
        )
        cache.uses_parameters = !isempty(cache.parameter_dependencies)
        _update_vector_constraint_cache!(model, outer_ci, cache, parameter_values)
        return
    end
    if outer_ci isa MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.Parameter}
        return
    end
    inner_ci = model.outer_to_inner_constraints[outer_ci]
    if outer_ci isa MOI.ConstraintIndex{MOI.VariableIndex}
        set = MOI.get(model.outer_model, MOI.ConstraintSet(), outer_ci)
        MOI.set(model.optimizer, MOI.ConstraintSet(), inner_ci, set)
        return
    end
    throw(MOI.UnsupportedError(MOI.ConstraintFunction()))
end

function _sync_objective_cache!(
    model::Optimizer{T};
    parameter_values::Dict{MOI.VariableIndex,T},
    force::Bool = false,
) where {T}
    outer = model.outer_model
    sense = MOI.get(outer, MOI.ObjectiveSense())
    MOI.set(model.optimizer, MOI.ObjectiveSense(), sense)
    if sense == MOI.FEASIBILITY_SENSE
        return
    end
    F = MOI.get(outer, MOI.ObjectiveFunctionType())
    F === nothing && return
    outer_function = MOI.get(outer, MOI.ObjectiveFunction{F}())
    if model.objective_cache === nothing
        inner_function = _prepare_objective_function(
            model,
            parameter_values,
            model.outer_to_inner_variables,
            outer_function,
        )
        MOI.set(model.optimizer, MOI.ObjectiveFunction{typeof(inner_function)}(), inner_function)
        parameter_dependencies = _function_parameter_dependencies(
            parameter_values,
            outer_function,
        )
        model.objective_cache = _ObjectiveCache{T}(
            outer_function,
            inner_function,
            !isempty(parameter_dependencies),
            parameter_dependencies,
        )
        return
    end
    cache = model.objective_cache
    cache.outer_function = outer_function
    cache.parameter_dependencies = _function_parameter_dependencies(
        parameter_values,
        outer_function,
    )
    cache.uses_parameters = !isempty(cache.parameter_dependencies)
    _update_objective_cache!(model, parameter_values; force)
    return
end

function _vector_affine_constant(::Type{T}, dimension::Integer) where {T}
    return MOI.VectorAffineFunction{T}(MOI.VectorAffineTerm{T}[], zeros(T, dimension))
end

function _vector_quadratic_constant(::Type{T}, dimension::Integer) where {T}
    return MOI.VectorQuadraticFunction{T}(
        MOI.VectorAffineTerm{T}[],
        MOI.VectorQuadraticTerm{T}[],
        zeros(T, dimension),
    )
end

_vector_objective_dimension(change::MOI.VectorConstantChange) = length(change.new_constant)

function _vector_objective_dimension(change::MOI.MultirowChange)
    isempty(change.new_coefficients) && throw(MOI.ModifyObjectiveNotAllowed(change))
    return maximum(first, change.new_coefficients)
end

function _objective_for_modify(
    ::Type{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    current,
    ::MOI.AbstractFunctionModification,
) where {T}
    if current === nothing
        return _scalar_constant(T, zero(T))
    elseif current isa MOI.ScalarAffineFunction{T}
        return current
    elseif current isa MOI.VariableIndex
        return convert(MOI.ScalarAffineFunction{T}, current)
    end
    return convert(MOI.ScalarAffineFunction{T}, current)
end

function _objective_for_modify(
    ::Type{T},
    ::Type{MOI.ScalarQuadraticFunction{T}},
    current,
    ::MOI.AbstractFunctionModification,
) where {T}
    if current === nothing
        return convert(MOI.ScalarQuadraticFunction{T}, _scalar_constant(T, zero(T)))
    elseif current isa MOI.ScalarQuadraticFunction{T}
        return current
    elseif current isa MOI.VariableIndex
        return convert(MOI.ScalarQuadraticFunction{T}, convert(MOI.ScalarAffineFunction{T}, current))
    elseif current isa MOI.ScalarAffineFunction{T}
        return convert(MOI.ScalarQuadraticFunction{T}, current)
    end
    return convert(MOI.ScalarQuadraticFunction{T}, current)
end

function _objective_for_modify(
    ::Type{T},
    ::Type{MOI.VectorAffineFunction{T}},
    current,
    change::Union{MOI.VectorConstantChange,MOI.MultirowChange},
) where {T}
    if current === nothing
        return _vector_affine_constant(T, _vector_objective_dimension(change))
    elseif current isa MOI.VectorAffineFunction{T}
        return current
    elseif current isa MOI.VectorOfVariables
        return convert(MOI.VectorAffineFunction{T}, current)
    end
    return convert(MOI.VectorAffineFunction{T}, current)
end

function _objective_for_modify(
    ::Type{T},
    ::Type{MOI.VectorQuadraticFunction{T}},
    current,
    change::Union{MOI.VectorConstantChange,MOI.MultirowChange},
) where {T}
    if current === nothing
        return _vector_quadratic_constant(T, _vector_objective_dimension(change))
    elseif current isa MOI.VectorQuadraticFunction{T}
        return current
    elseif current isa MOI.VectorOfVariables
        return convert(MOI.VectorQuadraticFunction{T}, convert(MOI.VectorAffineFunction{T}, current))
    elseif current isa MOI.VectorAffineFunction{T}
        return convert(MOI.VectorQuadraticFunction{T}, current)
    end
    return convert(MOI.VectorQuadraticFunction{T}, current)
end

function _objective_for_modify(
    ::Type{T},
    ::Type{F},
    _current,
    change::MOI.AbstractFunctionModification,
) where {T,F}
    throw(MOI.ModifyObjectiveNotAllowed(change))
end

function _modify_outer_objective!(
    model::Optimizer{T},
    attr::MOI.ObjectiveFunction{F},
    change::MOI.AbstractFunctionModification,
) where {T,F}
    current_type = MOI.get(model.outer_model, MOI.ObjectiveFunctionType())
    current =
        current_type === nothing ? nothing :
        MOI.get(model.outer_model, MOI.ObjectiveFunction{current_type}())
    base = try
        _objective_for_modify(T, F, current, change)
    catch err
        err isa MethodError && throw(MOI.ModifyObjectiveNotAllowed(change))
        rethrow()
    end
    new_objective = try
        MOIU.modify_function(base, change)
    catch err
        err isa MethodError && throw(MOI.ModifyObjectiveNotAllowed(change))
        rethrow()
    end
    MOI.set(model.outer_model, MOI.ObjectiveFunction{typeof(new_objective)}(), new_objective)
    return
end

function _build_objective_cache!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {T}
    outer = model.outer_model
    sense = MOI.get(outer, MOI.ObjectiveSense())
    MOI.set(model.optimizer, MOI.ObjectiveSense(), sense)
    model.objective_cache = nothing
    if sense == MOI.FEASIBILITY_SENSE
        return
    end
    F = MOI.get(outer, MOI.ObjectiveFunctionType())
    F === nothing && return
    outer_function = MOI.get(outer, MOI.ObjectiveFunction{F}())
    inner_function = _prepare_objective_function(
        model,
        parameter_values,
        variable_map,
        outer_function,
    )
    MOI.set(model.optimizer, MOI.ObjectiveFunction{typeof(inner_function)}(), inner_function)
    parameter_dependencies = _function_parameter_dependencies(
        parameter_values,
        outer_function,
    )
    model.objective_cache = _ObjectiveCache{T}(
        outer_function,
        inner_function,
        !isempty(parameter_dependencies),
        parameter_dependencies,
    )
    return
end

function _build_function_constraint_caches!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
) where {T}
    outer = model.outer_model
    empty!(model.scalar_constraint_caches)
    empty!(model.vector_constraint_caches)
    for (F, S) in MOI.get(outer, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        for outer_ci in MOI.get(outer, MOI.ListOfConstraintIndices{F,S}())
            outer_function = MOI.get(outer, MOI.ConstraintFunction(), outer_ci)
            outer_set = MOI.get(outer, MOI.ConstraintSet(), outer_ci)
            parameter_dependencies = _function_parameter_dependencies(
                parameter_values,
                outer_function,
            )
            label = _constraint_label(outer, outer_ci)
            inner_function, inner_set = _prepare_constraint_data(
                T,
                parameter_values,
                variable_map,
                outer_function,
                outer_set,
                label,
            )
            inner_ci = _add_prepared_constraint(model, inner_function, inner_set)
            constraint_map[outer_ci] = inner_ci
            _copy_constraint_attributes!(outer, model.optimizer, outer_ci, inner_ci)
            if inner_function isa MOI.AbstractScalarFunction && inner_set isa MOI.AbstractScalarSet
                model.scalar_constraint_caches[outer_ci] = _ScalarConstraintCache{T}(
                    outer_function,
                    outer_set,
                    inner_ci,
                    inner_function,
                    inner_set,
                    !isempty(parameter_dependencies),
                    parameter_dependencies,
                )
            else
                model.vector_constraint_caches[outer_ci] = _VectorConstraintCache(
                    outer_function,
                    outer_set,
                    inner_ci,
                    inner_function,
                    inner_set,
                    !isempty(parameter_dependencies),
                    parameter_dependencies,
                )
            end
        end
    end
    return
end

function _update_objective_cache!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T};
    force::Bool = false,
    changed_parameters::Set{MOI.VariableIndex} = Set{MOI.VariableIndex}(),
) where {T}
    cache = model.objective_cache
    cache === nothing && return
    if !force && !cache.uses_parameters
        return
    end
    if !force && !_depends_on_changed_parameter(cache.parameter_dependencies, changed_parameters)
        return
    end
    new_function = _prepare_objective_function(
        model,
        parameter_values,
        model.outer_to_inner_variables,
        cache.outer_function,
    )
    if new_function isa MOI.ScalarNonlinearFunction
        MOI.set(model.optimizer, MOI.ObjectiveFunction{typeof(new_function)}(), new_function)
        cache.current_function = new_function
        return
    end
    if typeof(new_function) != typeof(cache.current_function) ||
       cache.current_function isa MOI.ScalarNonlinearFunction
        MOI.set(model.optimizer, MOI.ObjectiveFunction{typeof(new_function)}(), new_function)
        cache.current_function = new_function
        return
    end
    target = MOI.ObjectiveFunction{typeof(cache.current_function)}()
    if new_function isa MOI.ScalarAffineFunction{T}
        _apply_scalar_affine_updates!(
            model.optimizer,
            target,
            cache.current_function,
            new_function,
        )
    else
        _apply_scalar_quadratic_updates!(
            model.optimizer,
            target,
            cache.current_function,
            new_function,
        )
    end
    cache.current_function = new_function
    return
end
