function _scalar_constant(::Type{T}, value) where {T}
    return MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{T}[], convert(T, value))
end

function _scalar_variable(::Type{T}, vi::MOI.VariableIndex) where {T}
    return MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm{T}[MOI.ScalarAffineTerm(one(T), vi)],
        zero(T),
    )
end

function _substitute_variable(
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    vi::MOI.VariableIndex,
) where {T}
    if haskey(parameter_values, vi)
        return _scalar_constant(T, parameter_values[vi])
    end
    return variable_map[vi]
end

function _collapse_scalar_function(f::MOI.ScalarAffineFunction{T}) where {T}
    return isempty(f.terms) ? MOI.constant(f) : MOIU.canonical(f)
end

function _collapse_scalar_function(f::MOI.ScalarQuadraticFunction{T}) where {T}
    if isempty(f.quadratic_terms)
        return _collapse_scalar_function(convert(MOI.ScalarAffineFunction{T}, f))
    end
    return MOIU.canonical(f)
end

_collapse_scalar_function(value::Real) = value

function _substitute_nonlinear_argument(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    arg,
) where {T}
    if arg isa MOI.VariableIndex
        return haskey(parameter_values, arg) ? parameter_values[arg] : variable_map[arg]
    elseif arg isa MOI.ScalarNonlinearFunction
        return MOI.ScalarNonlinearFunction(
            arg.head,
            Any[
                _substitute_nonlinear_argument(T, parameter_values, variable_map, child) for
                child in arg.args
            ],
        )
    elseif arg isa MOI.ScalarAffineFunction{<:Real}
        subbed = MOIU.substitute_variables(
            _substitute_variable_map(T, parameter_values, variable_map),
            arg,
        )
        return _collapse_scalar_function(subbed)
    elseif arg isa MOI.ScalarQuadraticFunction{<:Real}
        subbed = MOIU.substitute_variables(
            _substitute_variable_map(T, parameter_values, variable_map),
            arg,
        )
        return _collapse_scalar_function(subbed)
    elseif arg isa Real
        return convert(T, arg)
    end
    return arg
end

function _substitute_variable_map(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {T}
    return vi -> _substitute_variable(parameter_values, variable_map, vi)
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.VariableIndex,
) where {T}
    if haskey(parameter_values, func)
        return _scalar_constant(T, parameter_values[func])
    end
    return _scalar_variable(T, variable_map[func])
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.ScalarAffineFunction{<:Real},
) where {T}
    mapper = _substitute_variable_map(T, parameter_values, variable_map)
    return MOIU.canonical(MOIU.substitute_variables(mapper, func))
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.ScalarQuadraticFunction{<:Real},
) where {T}
    mapper = _substitute_variable_map(T, parameter_values, variable_map)
    return MOIU.canonical(MOIU.substitute_variables(mapper, func))
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.VectorOfVariables,
) where {T}
    variables = MOI.VariableIndex[]
    for vi in func.variables
        if haskey(parameter_values, vi)
            error("VectorOfVariables constraints cannot include parameters.")
        end
        push!(variables, variable_map[vi])
    end
    return MOI.VectorOfVariables(variables)
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.VectorAffineFunction{<:Real},
) where {T}
    mapper = _substitute_variable_map(T, parameter_values, variable_map)
    return MOIU.canonical(MOIU.substitute_variables(mapper, func))
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.VectorQuadraticFunction{<:Real},
) where {T}
    mapper = _substitute_variable_map(T, parameter_values, variable_map)
    return MOIU.canonical(MOIU.substitute_variables(mapper, func))
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.VectorNonlinearFunction,
) where {T}
    rows = [
        _transform_function(T, parameter_values, variable_map, row) for row in func.rows
    ]
    if all(row -> row isa MOI.ScalarAffineFunction{T}, rows)
        affine_rows = convert(Vector{MOI.ScalarAffineFunction{T}}, rows)
        maybe_variables = map(_vector_row_as_variable, affine_rows)
        if all(!isnothing, maybe_variables)
            return MOI.VectorOfVariables(MOI.VariableIndex[vi for vi in maybe_variables])
        end
        return MOI.VectorAffineFunction(affine_rows)
    end
    if all(row -> row isa Union{MOI.ScalarAffineFunction{T},MOI.ScalarQuadraticFunction{T}}, rows)
        quadratic_rows = MOI.ScalarQuadraticFunction{T}[
            row isa MOI.ScalarQuadraticFunction{T} ?
            row :
            convert(MOI.ScalarQuadraticFunction{T}, row) for row in rows
        ]
        return MOI.VectorQuadraticFunction(quadratic_rows)
    end
    nonlinear_rows = MOI.ScalarNonlinearFunction[
        _vector_nonlinear_row(T, row) for row in rows
    ]
    return MOI.VectorNonlinearFunction(nonlinear_rows)
end

function _normalize_scalar_result(::Type{T}, result) where {T}
    if result isa Real
        return _scalar_constant(T, result)
    elseif result isa MOI.VariableIndex
        return _scalar_variable(T, result)
    elseif result isa MOI.ScalarAffineFunction{T}
        return MOIU.canonical(result)
    elseif result isa MOI.ScalarAffineFunction
        return MOIU.canonical(convert(MOI.ScalarAffineFunction{T}, result))
    elseif result isa MOI.ScalarQuadraticFunction{T}
        return MOIU.canonical(result)
    elseif result isa MOI.ScalarQuadraticFunction
        return MOIU.canonical(convert(MOI.ScalarQuadraticFunction{T}, result))
    end
    return nothing
end

function _transform_function(
    ::Type{T},
    parameter_values::Dict{MOI.VariableIndex,T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    func::MOI.ScalarNonlinearFunction,
) where {T}
    substituted = _substitute_nonlinear_argument(T, parameter_values, variable_map, func)
    simplified = substituted isa MOI.ScalarNonlinearFunction ?
        MOI.Nonlinear.SymbolicAD.simplify(substituted) : substituted
    normalized = _normalize_scalar_result(T, simplified)
    return normalized === nothing ? simplified : normalized
end

function _vector_row_as_variable(
    row::MOI.ScalarAffineFunction{T},
) where {T}
    canonical = MOIU.canonical(row)
    if iszero(MOI.constant(canonical)) && length(canonical.terms) == 1
        term = only(canonical.terms)
        if isone(term.coefficient)
            return term.variable
        end
    end
    return nothing
end

function _vector_nonlinear_row(
    ::Type{T},
    row::MOI.ScalarAffineFunction{T},
) where {T}
    return convert(MOI.ScalarNonlinearFunction, row)
end

function _vector_nonlinear_row(
    ::Type{T},
    row::MOI.ScalarQuadraticFunction{T},
) where {T}
    return convert(MOI.ScalarNonlinearFunction, row)
end

function _vector_nonlinear_row(
    ::Type{T},
    row::MOI.ScalarNonlinearFunction,
) where {T}
    return row
end
