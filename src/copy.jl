function _copy_constraint_name!(
    outer_model,
    inner_model,
    outer_ci::MOI.ConstraintIndex,
    inner_ci::MOI.ConstraintIndex,
)
    name = MOI.get(outer_model, MOI.ConstraintName(), outer_ci)
    if !isempty(name) &&
       MOI.supports(inner_model, MOI.ConstraintName(), typeof(inner_ci))
        try
            MOI.set(inner_model, MOI.ConstraintName(), inner_ci, name)
        catch err
            err isa MOI.UnsupportedError && return
            err isa MOI.NotAllowedError && return
            rethrow()
        end
    end
    return
end

function _copy_variable_name!(
    outer_model,
    inner_model,
    outer_vi::MOI.VariableIndex,
    inner_vi::MOI.VariableIndex,
)
    name = MOI.get(outer_model, MOI.VariableName(), outer_vi)
    if !isempty(name) && MOI.supports(inner_model, MOI.VariableName(), MOI.VariableIndex)
        try
            MOI.set(inner_model, MOI.VariableName(), inner_vi, name)
        catch err
            err isa MOI.UnsupportedError && return
            err isa MOI.NotAllowedError && return
            rethrow()
        end
    end
    return
end

function _copy_variable_attributes!(
    outer_model,
    inner_model,
    outer_vi::MOI.VariableIndex,
    inner_vi::MOI.VariableIndex,
)
    _copy_variable_name!(outer_model, inner_model, outer_vi, inner_vi)
    for attr in MOI.get(outer_model, MOI.ListOfVariableAttributesSet())
        attr == MOI.VariableName() && continue
        MOI.is_set_by_optimize(attr) && continue
        MOI.supports(inner_model, attr, MOI.VariableIndex) || continue
        value = MOI.get(outer_model, attr, outer_vi)
        value === nothing && continue
        try
            MOI.set(inner_model, attr, inner_vi, value)
        catch err
            err isa MOI.UnsupportedError && continue
            err isa MOI.NotAllowedError && continue
            rethrow()
        end
    end
    return
end

function _constraint_label(outer_model, ci::MOI.ConstraintIndex)
    name = MOI.get(outer_model, MOI.ConstraintName(), ci)
    return isempty(name) ? string(ci) : name
end

function _copy_constraint_attributes!(
    outer_model,
    inner_model,
    outer_ci::MOI.ConstraintIndex,
    inner_ci::MOI.ConstraintIndex,
)
    _copy_constraint_name!(outer_model, inner_model, outer_ci, inner_ci)
    outer_function = MOI.get(outer_model, MOI.ConstraintFunction(), outer_ci)
    outer_set = MOI.get(outer_model, MOI.ConstraintSet(), outer_ci)
    for attr in MOI.get(
        outer_model,
        MOI.ListOfConstraintAttributesSet{typeof(outer_function),typeof(outer_set)}(),
    )
        attr == MOI.ConstraintName() && continue
        attr isa Union{MOI.ConstraintFunction,MOI.ConstraintSet} && continue
        MOI.is_set_by_optimize(attr) && continue
        MOI.supports(inner_model, attr, typeof(inner_ci)) || continue
        value = MOI.get(outer_model, attr, outer_ci)
        value === nothing && continue
        try
            MOI.set(inner_model, attr, inner_ci, value)
        catch err
            err isa MOI.UnsupportedError && continue
            err isa MOI.NotAllowedError && continue
            rethrow()
        end
    end
    return
end

_model_attribute_priority(::MOI.UserDefinedFunction) = 0.0
_model_attribute_priority(::MOI.ObjectiveSense) = 10.0
_model_attribute_priority(::MOI.ObjectiveFunction) = 20.0
_model_attribute_priority(::MOI.AbstractModelAttribute) = 30.0

function _index_map(
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
)
    index_map = MOIU.IndexMap()
    for (outer_vi, inner_vi) in variable_map
        index_map[outer_vi] = inner_vi
    end
    for (outer_ci, inner_ci) in constraint_map
        index_map[outer_ci] = inner_ci
    end
    return index_map
end

function _copy_model_attributes!(
    model::Optimizer,
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
)
    attrs = MOI.get(model.outer_model, MOI.ListOfModelAttributesSet())
    sort!(attrs; by = _model_attribute_priority)
    index_map = _index_map(variable_map, constraint_map)
    for attr in attrs
        attr isa Union{MOI.ObjectiveSense,MOI.ObjectiveFunction} && continue
        MOI.is_set_by_optimize(attr) && continue
        MOI.supports(model.optimizer, attr) || continue
        value = MOI.get(model.outer_model, attr)
        value === nothing && continue
        try
            MOI.set(model.optimizer, attr, MOIU.map_indices(index_map, attr, value))
        catch err
            err isa MOI.UnsupportedAttribute && continue
            err isa MOI.UnsupportedError && continue
            err isa MOI.NotAllowedError && continue
            rethrow()
        end
    end
    return
end

function _copy_variables!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
) where {T}
    variable_map = Dict{MOI.VariableIndex,MOI.VariableIndex}()
    for vi in MOI.get(model.outer_model, MOI.ListOfVariableIndices())
        if haskey(parameter_values, vi)
            continue
        end
        inner_vi = MOI.add_variable(model.optimizer)
        variable_map[vi] = inner_vi
        _copy_variable_attributes!(model.outer_model, model.optimizer, vi, inner_vi)
    end
    return variable_map
end

function _copy_variable_constraints!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    constraint_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
) where {T}
    outer = model.outer_model
    for (F, S) in MOI.get(outer, MOI.ListOfConstraintTypesPresent())
        F != MOI.VariableIndex && continue
        for outer_ci in MOI.get(outer, MOI.ListOfConstraintIndices{F,S}())
            outer_vi = MOI.get(outer, MOI.ConstraintFunction(), outer_ci)
            set = MOI.get(outer, MOI.ConstraintSet(), outer_ci)
            if set isa MOI.Parameter{T}
                continue
            elseif haskey(parameter_values, outer_vi)
                error("Parameter variables cannot participate in non-parameter variable constraints.")
            end
            inner_ci = MOI.add_constraint(model.optimizer, variable_map[outer_vi], set)
            constraint_map[outer_ci] = inner_ci
            _copy_constraint_attributes!(outer, model.optimizer, outer_ci, inner_ci)
        end
    end
    return
end

function _prepare_objective_function(
    ::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.AbstractScalarFunction,
) where {T}
    return _transform_function(T, parameter_values, variable_map, func)
end

function _prepare_constraint_data(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func,
    set,
    label::AbstractString,
) where {T}
    transformed = _transform_function(T, parameter_values, variable_map, func)
    if transformed isa MOI.AbstractScalarFunction &&
       !(transformed isa MOI.ScalarNonlinearFunction) &&
       set isa MOI.AbstractScalarSet
        normalized_func, normalized_set = MOIU.normalize_constant(transformed, set)
        return MOIU.canonical(normalized_func), normalized_set
    end
    return transformed, set
end

function _add_prepared_constraint(model::Optimizer, func, set)
    return MOI.add_constraint(model.optimizer, func, set)
end
