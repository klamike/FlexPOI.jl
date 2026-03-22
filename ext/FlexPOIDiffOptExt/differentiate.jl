function DiffOpt.forward_differentiate!(model::FlexPOI.Optimizer{T}) where {T}
    _require_diffopt_inner(model, DiffOpt.forward_differentiate!)
    FlexPOI._require_result(model, DiffOpt.ForwardVariablePrimal())
    DiffOpt.empty_input_sensitivities!(model.optimizer)
    parameter_values = FlexPOI._parameter_values(model)
    variable_map = model.outer_to_inner_variables
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
            if perturbation !== nothing
                MOI.set(model.optimizer, DiffOpt.ForwardObjectiveFunction(), perturbation)
            end
        end
    end

    for (F, S) in MOI.get(outer, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        for outer_ci in MOI.get(outer, MOI.ListOfConstraintIndices{F,S}())
            context = "constraint " * FlexPOI._constraint_label(outer, outer_ci)
            func = MOI.get(outer, MOI.ConstraintFunction(), outer_ci)
            perturbation = _build_forward_perturbation(
                model,
                func,
                parameter_values,
                variable_map,
                context,
            )
            perturbation === nothing && continue
            if perturbation isa MOI.ScalarQuadraticFunction
                error(
                    "DiffOpt parameter perturbations for constraints must be affine. " *
                    "$context simplified to a quadratic perturbation.",
                )
            end
            inner_ci = model.outer_to_inner_constraints[outer_ci]
            MOI.set(
                model.optimizer,
                DiffOpt.ForwardConstraintFunction(),
                inner_ci,
                perturbation,
            )
        end
    end

    DiffOpt.forward_differentiate!(model.optimizer)
    return
end

function DiffOpt.reverse_differentiate!(model::FlexPOI.Optimizer{T}) where {T}
    _require_diffopt_inner(model, DiffOpt.reverse_differentiate!)
    FlexPOI._require_result(model, DiffOpt.ReverseObjectiveSensitivity())
    DiffOpt.reverse_differentiate!(model.optimizer)

    sensitivity_data = _get_sensitivity_data(model)
    empty!(sensitivity_data.parameter_output_backward)

    parameter_values = FlexPOI._parameter_values(model)
    variable_map = model.outer_to_inner_variables
    outer = model.outer_model

    sense = MOI.get(outer, MOI.ObjectiveSense())
    objective_gradient = nothing
    objective_function = nothing
    if sense != MOI.FEASIBILITY_SENSE
        F = MOI.get(outer, MOI.ObjectiveFunctionType())
        if F !== nothing
            objective_function = MOI.get(outer, MOI.ObjectiveFunction{F}())
            objective_gradient = MOI.get(model.optimizer, DiffOpt.ReverseObjectiveFunction())
        end
    end

    for parameter in _parameter_variables(model)
        value = zero(T)
        if objective_gradient !== nothing
            partial = _parameter_partial_derivative(
                model,
                objective_function,
                parameter,
                parameter_values,
                variable_map,
                "objective",
            )
            if partial !== nothing
                value += _function_inner_product(T, objective_gradient, partial)
            end
        end
        for (F, S) in MOI.get(outer, MOI.ListOfConstraintTypesPresent())
            F == MOI.VariableIndex && continue
            for outer_ci in MOI.get(outer, MOI.ListOfConstraintIndices{F,S}())
                func = MOI.get(outer, MOI.ConstraintFunction(), outer_ci)
                context = "constraint " * FlexPOI._constraint_label(outer, outer_ci)
                partial = _parameter_partial_derivative(
                    model,
                    func,
                    parameter,
                    parameter_values,
                    variable_map,
                    context,
                )
                partial === nothing && continue
                gradient = MOI.get(
                    model.optimizer,
                    DiffOpt.ReverseConstraintFunction(),
                    model.outer_to_inner_constraints[outer_ci],
                )
                value += _function_inner_product(T, gradient, partial)
            end
        end
        sensitivity_data.parameter_output_backward[parameter] = value
    end
    return
end
