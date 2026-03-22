function _require_result(model::Optimizer, attr)
    if !model.result_available
        throw(MOI.GetAttributeNotAllowed(attr, "No result is available. Call optimize! first."))
    end
    return
end

function _require_conflict(model::Optimizer, attr)
    if !model.conflict_available
        throw(
            MOI.GetAttributeNotAllowed(
                attr,
                "No conflict is available. Call compute_conflict! first.",
            ),
        )
    end
    return
end

function _sync_parameter_state!(model::Optimizer{T}) where {T}
    existing_parameters = Dict{MOI.VariableIndex,T}()
    existing_constraints = Dict{
        MOI.VariableIndex,
        MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    }()
    for ci in MOI.get(
        model.outer_model,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Parameter{T}}(),
    )
        vi = MOI.get(model.outer_model, MOI.ConstraintFunction(), ci)
        set = MOI.get(model.outer_model, MOI.ConstraintSet(), ci)
        existing_parameters[vi] = set.value
        existing_constraints[vi] = ci
    end
    model.parameter_values = existing_parameters
    model.parameter_constraints = existing_constraints
    filter!(pair -> haskey(existing_parameters, pair.first), model.pending_parameter_values)
    return
end

function _parameter_values(model::Optimizer{T}) where {T}
    values = copy(model.parameter_values)
    for (vi, value) in model.pending_parameter_values
        values[vi] = value
    end
    return values
end

function _parameter_value(model::Optimizer{T}, vi::MOI.VariableIndex) where {T}
    if haskey(model.pending_parameter_values, vi)
        return model.pending_parameter_values[vi]
    end
    haskey(model.parameter_values, vi) && return model.parameter_values[vi]
    error("Variable $vi is not a parameter.")
end

function _is_parameter(model::Optimizer{T}, vi::MOI.VariableIndex) where {T}
    return haskey(model.parameter_constraints, vi)
end

function _same_parameter_values(left::Dict, right::Dict)
    length(left) == length(right) || return false
    for (key, value) in left
        haskey(right, key) || return false
        isequal(right[key], value) || return false
    end
    return true
end

function _changed_parameter_variables(
    left::Dict{MOI.VariableIndex,T},
    right::Dict{MOI.VariableIndex,T},
) where {T}
    changed = Set{MOI.VariableIndex}()
    for (key, value) in left
        if !haskey(right, key) || !isequal(right[key], value)
            push!(changed, key)
        end
    end
    for key in keys(right)
        if !haskey(left, key)
            push!(changed, key)
        end
    end
    return changed
end

function _depends_on_changed_parameter(
    dependencies::Vector{MOI.VariableIndex},
    changed::Set{MOI.VariableIndex},
)
    return any(variable -> in(variable, changed), dependencies)
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable::MOI.VariableIndex,
) where {T}
    if haskey(parameter_values, variable)
        push!(dependencies, variable)
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarAffineFunction,
) where {T}
    for term in func.terms
        _collect_parameter_dependencies!(dependencies, parameter_values, term.variable)
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarQuadraticFunction,
) where {T}
    for term in func.affine_terms
        _collect_parameter_dependencies!(dependencies, parameter_values, term.variable)
    end
    for term in func.quadratic_terms
        _collect_parameter_dependencies!(dependencies, parameter_values, term.variable_1)
        _collect_parameter_dependencies!(dependencies, parameter_values, term.variable_2)
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorOfVariables,
) where {T}
    for variable in func.variables
        _collect_parameter_dependencies!(dependencies, parameter_values, variable)
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorAffineFunction,
) where {T}
    for term in func.terms
        _collect_parameter_dependencies!(
            dependencies,
            parameter_values,
            term.scalar_term.variable,
        )
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorQuadraticFunction,
) where {T}
    for term in func.affine_terms
        _collect_parameter_dependencies!(
            dependencies,
            parameter_values,
            term.scalar_term.variable,
        )
    end
    for term in func.quadratic_terms
        _collect_parameter_dependencies!(
            dependencies,
            parameter_values,
            term.scalar_term.variable_1,
        )
        _collect_parameter_dependencies!(
            dependencies,
            parameter_values,
            term.scalar_term.variable_2,
        )
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarNonlinearFunction,
) where {T}
    for arg in func.args
        _collect_parameter_dependencies!(dependencies, parameter_values, arg)
    end
    return
end

function _collect_parameter_dependencies!(
    dependencies::Set{MOI.VariableIndex},
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorNonlinearFunction,
) where {T}
    for row in func.rows
        _collect_parameter_dependencies!(dependencies, parameter_values, row)
    end
    return
end

function _collect_parameter_dependencies!(
    ::Set{MOI.VariableIndex},
    ::Dict{MOI.VariableIndex,T},
    ::Any,
) where {T}
    return
end

function _function_parameter_dependencies(
    parameter_values::Dict{MOI.VariableIndex,T},
    func,
) where {T}
    dependencies = Set{MOI.VariableIndex}()
    _collect_parameter_dependencies!(dependencies, parameter_values, func)
    result = collect(dependencies)
    sort!(result; by = variable -> variable.value)
    return result
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    vi::MOI.VariableIndex,
) where {T}
    return haskey(parameter_values, vi)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarAffineFunction,
) where {T}
    return any(haskey(parameter_values, term.variable) for term in func.terms)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarQuadraticFunction,
) where {T}
    return any(
        haskey(parameter_values, term.variable_1) ||
        haskey(parameter_values, term.variable_2) for term in func.quadratic_terms
    ) || any(haskey(parameter_values, term.variable) for term in func.affine_terms)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorOfVariables,
) where {T}
    return any(haskey(parameter_values, vi) for vi in func.variables)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorAffineFunction,
) where {T}
    return any(haskey(parameter_values, term.scalar_term.variable) for term in func.terms)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorQuadraticFunction,
) where {T}
    return any(
        haskey(parameter_values, term.scalar_term.variable_1) ||
        haskey(parameter_values, term.scalar_term.variable_2) for term in func.quadratic_terms
    ) || any(haskey(parameter_values, term.scalar_term.variable) for term in func.affine_terms)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.ScalarNonlinearFunction,
) where {T}
    return any(_function_uses_parameters(parameter_values, arg) for arg in func.args)
end

function _function_uses_parameters(
    parameter_values::Dict{MOI.VariableIndex,T},
    func::MOI.VectorNonlinearFunction,
) where {T}
    return any(_function_uses_parameters(parameter_values, row) for row in func.rows)
end

function _function_uses_parameters(
    ::Dict{MOI.VariableIndex,T},
    ::Any,
) where {T}
    return false
end

function _commit_parameter_values!(
    model::Optimizer{T},
    parameter_values::Dict{MOI.VariableIndex,T},
) where {T}
    model.parameter_values = copy(parameter_values)
    empty!(model.pending_parameter_values)
    return
end

function _should_retry_full_rebuild(err)
    return err isa MOI.UnsupportedError || err isa MOI.NotAllowedError
end
