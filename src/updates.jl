function _affine_coefficients(func::MOI.ScalarAffineFunction{T}) where {T}
    coefficients = Dict{MOI.VariableIndex,T}()
    for term in func.terms
        coefficients[term.variable] = get(coefficients, term.variable, zero(T)) + term.coefficient
    end
    return coefficients
end

function _affine_coefficients(func::MOI.ScalarQuadraticFunction{T}) where {T}
    coefficients = Dict{MOI.VariableIndex,T}()
    for term in func.affine_terms
        coefficients[term.variable] = get(coefficients, term.variable, zero(T)) + term.coefficient
    end
    return coefficients
end

function _quadratic_coefficients(func::MOI.ScalarQuadraticFunction{T}) where {T}
    coefficients = Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T}()
    for term in func.quadratic_terms
        pair = term.variable_1.value <= term.variable_2.value ?
            (term.variable_1, term.variable_2) :
            (term.variable_2, term.variable_1)
        coefficients[pair] = get(coefficients, pair, zero(T)) + term.coefficient
    end
    return coefficients
end

function _apply_scalar_affine_updates!(inner_model, target, current, new)
    T = typeof(MOI.constant(new))
    current_coefficients = _affine_coefficients(current)
    new_coefficients = _affine_coefficients(new)
    variables = Set{MOI.VariableIndex}(keys(current_coefficients))
    union!(variables, keys(new_coefficients))
    for variable in variables
        current_value = get(current_coefficients, variable, zero(T))
        new_value = get(new_coefficients, variable, zero(T))
        if !isequal(current_value, new_value)
            MOI.modify(inner_model, target, MOI.ScalarCoefficientChange(variable, new_value))
        end
    end
    if !isequal(MOI.constant(current), MOI.constant(new))
        MOI.modify(inner_model, target, MOI.ScalarConstantChange(MOI.constant(new)))
    end
    return
end

function _apply_scalar_quadratic_updates!(inner_model, target, current, new)
    T = typeof(MOI.constant(new))
    current_affine = _affine_coefficients(current)
    new_affine = _affine_coefficients(new)
    affine_variables = Set{MOI.VariableIndex}(keys(current_affine))
    union!(affine_variables, keys(new_affine))
    for variable in affine_variables
        current_value = get(current_affine, variable, zero(T))
        new_value = get(new_affine, variable, zero(T))
        if !isequal(current_value, new_value)
            MOI.modify(inner_model, target, MOI.ScalarCoefficientChange(variable, new_value))
        end
    end
    current_quadratic = _quadratic_coefficients(current)
    new_quadratic = _quadratic_coefficients(new)
    quadratic_pairs = Set{Tuple{MOI.VariableIndex,MOI.VariableIndex}}(keys(current_quadratic))
    union!(quadratic_pairs, keys(new_quadratic))
    for (variable_1, variable_2) in quadratic_pairs
        current_value = get(current_quadratic, (variable_1, variable_2), zero(T))
        new_value = get(new_quadratic, (variable_1, variable_2), zero(T))
        if !isequal(current_value, new_value)
            MOI.modify(
                inner_model,
                target,
                MOI.ScalarQuadraticCoefficientChange(variable_1, variable_2, new_value),
            )
        end
    end
    if !isequal(MOI.constant(current), MOI.constant(new))
        MOI.modify(inner_model, target, MOI.ScalarConstantChange(MOI.constant(new)))
    end
    return
end

function _vector_affine_coefficients(func::MOI.VectorAffineFunction{T}) where {T}
    coefficients = Dict{MOI.VariableIndex,Dict{Int,T}}()
    for term in func.terms
        by_row = get!(coefficients, term.scalar_term.variable, Dict{Int,T}())
        row = term.output_index
        by_row[row] = get(by_row, row, zero(T)) + term.scalar_term.coefficient
    end
    return coefficients
end

function _vector_affine_coefficients(func::MOI.VectorQuadraticFunction{T}) where {T}
    coefficients = Dict{MOI.VariableIndex,Dict{Int,T}}()
    for term in func.affine_terms
        by_row = get!(coefficients, term.scalar_term.variable, Dict{Int,T}())
        row = term.output_index
        by_row[row] = get(by_row, row, zero(T)) + term.scalar_term.coefficient
    end
    return coefficients
end

function _vector_quadratic_coefficients(func::MOI.VectorQuadraticFunction{T}) where {T}
    coefficients = Dict{Tuple{Int,MOI.VariableIndex,MOI.VariableIndex},T}()
    for term in func.quadratic_terms
        v1, v2 = term.scalar_term.variable_1, term.scalar_term.variable_2
        pair = v1.value <= v2.value ? (term.output_index, v1, v2) : (term.output_index, v2, v1)
        coefficients[pair] = get(coefficients, pair, zero(T)) + term.scalar_term.coefficient
    end
    return coefficients
end

function _apply_vector_constant_update!(inner_model, target, current_constants, new_constants)
    if !isequal(current_constants, new_constants)
        MOI.modify(inner_model, target, MOI.VectorConstantChange(copy(new_constants)))
    end
    return
end

function _apply_multirow_updates!(inner_model, target, current, new, ::Type{T}) where {T}
    variables = Set{MOI.VariableIndex}(keys(current))
    union!(variables, keys(new))
    for variable in variables
        current_rows = get(current, variable, Dict{Int,T}())
        new_rows = get(new, variable, Dict{Int,T}())
        rows = Set{Int}(keys(current_rows))
        union!(rows, keys(new_rows))
        changes = Tuple{Int,T}[]
        for row in sort!(collect(rows))
            current_value = get(current_rows, row, zero(T))
            new_value = get(new_rows, row, zero(T))
            if !isequal(current_value, new_value)
                push!(changes, (row, new_value))
            end
        end
        isempty(changes) && continue
        MOI.modify(inner_model, target, MOI.MultirowChange(variable, changes))
    end
    return
end

function _apply_vector_affine_updates!(inner_model, target, current, new)
    T = eltype(new.constants)
    _apply_vector_constant_update!(inner_model, target, current.constants, new.constants)
    _apply_multirow_updates!(
        inner_model,
        target,
        _vector_affine_coefficients(current),
        _vector_affine_coefficients(new),
        T,
    )
    return
end

function _apply_vector_quadratic_updates!(inner_model, target, current, new)
    T = eltype(new.constants)
    _apply_vector_constant_update!(inner_model, target, current.constants, new.constants)
    _apply_multirow_updates!(
        inner_model,
        target,
        _vector_affine_coefficients(current),
        _vector_affine_coefficients(new),
        T,
    )
    return
end
