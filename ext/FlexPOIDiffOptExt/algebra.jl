function _standard_form(func)
    return applicable(DiffOpt.standard_form, func) ? DiffOpt.standard_form(func) : func
end

function _as_scalar_quadratic(::Type{T}, func) where {T}
    func = _standard_form(func)
    if func === nothing || func === false
        return nothing
    elseif func === true
        return convert(
            MOI.ScalarQuadraticFunction{T},
            FlexPOI._scalar_constant(T, one(T)),
        )
    elseif func isa Real
        return convert(
            MOI.ScalarQuadraticFunction{T},
            FlexPOI._scalar_constant(T, func),
        )
    elseif func isa MOI.VariableIndex
        return convert(
            MOI.ScalarQuadraticFunction{T},
            FlexPOI._scalar_variable(T, func),
        )
    elseif func isa MOI.ScalarAffineFunction{T}
        return MOIU.canonical(convert(MOI.ScalarQuadraticFunction{T}, func))
    elseif func isa MOI.ScalarAffineFunction
        converted = convert(MOI.ScalarAffineFunction{T}, func)
        return MOIU.canonical(convert(MOI.ScalarQuadraticFunction{T}, converted))
    elseif func isa MOI.ScalarQuadraticFunction{T}
        return MOIU.canonical(func)
    elseif func isa MOI.ScalarQuadraticFunction
        return MOIU.canonical(convert(MOI.ScalarQuadraticFunction{T}, func))
    end
    error("Unsupported scalar function in DiffOpt sensitivity contraction: $(typeof(func))")
end

function _affine_coefficients(
    ::Type{T},
    func::MOI.ScalarQuadraticFunction{T},
) where {T}
    coeffs = Dict{MOI.VariableIndex,T}()
    for term in func.affine_terms
        coeffs[term.variable] = get(coeffs, term.variable, zero(T)) + term.coefficient
    end
    return coeffs
end

function _quadratic_coefficients(
    ::Type{T},
    func::MOI.ScalarQuadraticFunction{T},
) where {T}
    coeffs = Dict{Tuple{MOI.VariableIndex,MOI.VariableIndex},T}()
    for term in func.quadratic_terms
        pair = term.variable_1.value <= term.variable_2.value ?
            (term.variable_1, term.variable_2) :
            (term.variable_2, term.variable_1)
        coeffs[pair] = get(coeffs, pair, zero(T)) + term.coefficient
    end
    return coeffs
end

function _scalar_inner_product(::Type{T}, left, right) where {T}
    left_q = _as_scalar_quadratic(T, left)
    right_q = _as_scalar_quadratic(T, right)
    if left_q === nothing || right_q === nothing
        return zero(T)
    end
    value = MOI.constant(left_q) * MOI.constant(right_q)
    left_aff = _affine_coefficients(T, left_q)
    right_aff = _affine_coefficients(T, right_q)
    for (variable, coefficient) in left_aff
        value += coefficient * get(right_aff, variable, zero(T))
    end
    left_quad = _quadratic_coefficients(T, left_q)
    right_quad = _quadratic_coefficients(T, right_q)
    for (pair, coefficient) in left_quad
        value += coefficient * get(right_quad, pair, zero(T))
    end
    return value
end

function _function_inner_product(::Type{T}, left, right) where {T}
    left = _standard_form(left)
    right = _standard_form(right)
    if left isa MOI.AbstractVectorFunction || right isa MOI.AbstractVectorFunction
        left_rows = left isa MOI.AbstractVectorFunction ? MOIU.scalarize(left) : left
        right_rows = right isa MOI.AbstractVectorFunction ? MOIU.scalarize(right) : right
        @assert length(left_rows) == length(right_rows)
        value = zero(T)
        for (left_row, right_row) in zip(left_rows, right_rows)
            value += _scalar_inner_product(T, left_row, right_row)
        end
        return value
    end
    return _scalar_inner_product(T, left, right)
end

function _inner_primal_values(model::FlexPOI.Optimizer{T}) where {T}
    primal_values = Dict{MOI.VariableIndex,T}()
    for inner_vi in Base.values(model.outer_to_inner_variables)
        primal_values[inner_vi] = MOI.get(model.optimizer, MOI.VariablePrimal(), inner_vi)
    end
    return primal_values
end

function _evaluate_scalar_function(
    ::Type{T},
    func::MOI.ScalarAffineFunction{T},
    values::Dict{MOI.VariableIndex,T},
) where {T}
    value = MOI.constant(func)
    for term in func.terms
        value += term.coefficient * values[term.variable]
    end
    return value
end

function _evaluate_scalar_function(
    ::Type{T},
    func::MOI.ScalarQuadraticFunction{T},
    values::Dict{MOI.VariableIndex,T},
) where {T}
    return MOIU.eval_variables(func) do vi
        values[vi]
    end
end

function _objective_gradient(
    ::Type{T},
    func::MOI.ScalarAffineFunction{T},
    values::Dict{MOI.VariableIndex,T},
) where {T}
    gradient = Dict{MOI.VariableIndex,T}()
    for term in func.terms
        gradient[term.variable] = get(gradient, term.variable, zero(T)) + term.coefficient
    end
    return gradient
end

function _objective_gradient(
    ::Type{T},
    func::MOI.ScalarQuadraticFunction{T},
    values::Dict{MOI.VariableIndex,T},
) where {T}
    gradient = Dict{MOI.VariableIndex,T}()
    for term in func.affine_terms
        gradient[term.variable] = get(gradient, term.variable, zero(T)) + term.coefficient
    end
    for term in func.quadratic_terms
        if term.variable_1 == term.variable_2
            gradient[term.variable_1] =
                get(gradient, term.variable_1, zero(T)) +
                term.coefficient * values[term.variable_1]
        else
            gradient[term.variable_1] =
                get(gradient, term.variable_1, zero(T)) +
                term.coefficient * values[term.variable_2]
            gradient[term.variable_2] =
                get(gradient, term.variable_2, zero(T)) +
                term.coefficient * values[term.variable_1]
        end
    end
    return gradient
end

function _forward_objective_sensitivity_fallback(
    model::FlexPOI.Optimizer{T},
) where {T}
    FlexPOI._require_result(model, DiffOpt.ForwardObjectiveSensitivity())
    parameter_values = FlexPOI._parameter_values(model)
    variable_map = model.outer_to_inner_variables
    inner_values = _inner_primal_values(model)

    direct = zero(T)
    outer = model.outer_model
    sense = MOI.get(outer, MOI.ObjectiveSense())
    if sense != MOI.FEASIBILITY_SENSE
        F = MOI.get(outer, MOI.ObjectiveFunctionType())
        if F !== nothing
            objective = MOI.get(outer, MOI.ObjectiveFunction{F}())
            perturbation = _build_forward_perturbation(
                model,
                objective,
                parameter_values,
                variable_map,
                "objective",
            )
            if perturbation isa MOI.ScalarAffineFunction{T}
                direct = _evaluate_scalar_function(T, perturbation, inner_values)
            elseif perturbation isa MOI.ScalarQuadraticFunction{T}
                direct = _evaluate_scalar_function(T, perturbation, inner_values)
            elseif perturbation !== nothing
                error(
                    "Unsupported DiffOpt objective perturbation type in fallback: " *
                    string(typeof(perturbation)),
                )
            end
        end
    end

    F_inner = MOI.get(model.optimizer, MOI.ObjectiveFunctionType())
    if F_inner === nothing
        return direct
    end
    current_objective = MOI.get(model.optimizer, MOI.ObjectiveFunction{F_inner}())
    current_objective = _standard_form(current_objective)
    current_objective_q = _as_scalar_quadratic(T, current_objective)
    current_objective_q === nothing && return direct

    gradient = _objective_gradient(T, current_objective_q, inner_values)
    indirect = zero(T)
    for (inner_vi, coefficient) in gradient
        dx = MOI.get(model.optimizer, DiffOpt.ForwardVariablePrimal(), inner_vi)
        indirect += coefficient * dx
    end
    return direct + indirect
end
