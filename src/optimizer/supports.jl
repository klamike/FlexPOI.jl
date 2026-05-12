_storage_model(model::Optimizer) = model.outer_model.model

function _supports_outer_model_attribute(model::Optimizer, attr::MOI.AbstractModelAttribute)
    if attr isa MOI.ListOfSupportedNonlinearOperators
        return true
    elseif MOI.is_copyable(attr)
        return MOI.supports(_storage_model(model), attr)
    end
    return false
end

function MOI.supports(model::Optimizer, attr::MOI.AbstractModelAttribute)
    if MOI.is_set_by_optimize(attr)
        return MOI.supports(model.optimizer, attr)
    elseif MOI.is_copyable(attr)
        return _supports_outer_model_attribute(model, attr) ||
               MOI.supports(model.optimizer, attr)
    end
    return _supports_outer_model_attribute(model, attr)
end

function MOI.supports(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.supports(model.optimizer, attr)
end

function _supports_outer_variable_attribute(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    index_type::Type{MOI.VariableIndex},
)
    if MOI.is_copyable(attr)
        return MOI.supports(_storage_model(model), attr, index_type)
    elseif attr isa MOI.VariablePrimal
        return true
    end
    return false
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    index_type::Type{MOI.VariableIndex},
)
    if MOI.is_set_by_optimize(attr)
        return MOI.supports(model.optimizer, attr, index_type)
    elseif MOI.is_copyable(attr)
        return _supports_outer_variable_attribute(model, attr, index_type) ||
               MOI.supports(model.optimizer, attr, index_type)
    end
    return _supports_outer_variable_attribute(model, attr, index_type)
end

function _supports_outer_constraint_attribute(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    index_type::Type{<:MOI.ConstraintIndex},
)
    if MOI.is_copyable(attr)
        return MOI.supports(_storage_model(model), attr, index_type)
    elseif attr isa Union{
        MOI.ConstraintFunction,
        MOI.ConstraintSet,
        MOI.CanonicalConstraintFunction,
        MOI.ConstraintPrimal,
    }
        return true
    elseif attr isa MOI.ConstraintConflictStatus
        return true
    end
    return false
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    index_type::Type{<:MOI.ConstraintIndex},
)
    if MOI.is_set_by_optimize(attr)
        return MOI.supports(model.optimizer, attr, index_type)
    elseif MOI.is_copyable(attr)
        return _supports_outer_constraint_attribute(model, attr, index_type) ||
               MOI.supports(model.optimizer, attr, index_type)
    end
    return _supports_outer_constraint_attribute(model, attr, index_type)
end

function _validate_copyable_attributes(dest::Optimizer, src::MOI.ModelLike)
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        MOI.supports(dest, attr) || throw(MOI.UnsupportedAttribute(attr))
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        MOI.supports(dest, attr, MOI.VariableIndex) || throw(MOI.UnsupportedAttribute(attr))
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        index_type = MOI.ConstraintIndex{F,S}
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            MOI.supports(dest, attr, index_type) || throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return
end

function MOI.supports_add_constrained_variable(model::Optimizer, set)
    return MOI.supports_add_constrained_variable(model, typeof(set))
end

function MOI.supports_add_constrained_variable(
    model::Optimizer,
    set_type::Type{<:MOI.AbstractScalarSet},
)
    return MOI.supports_add_constrained_variable(_storage_model(model), set_type)
end

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{<:MOI.Parameter},
)
    return true
end

function MOI.supports_add_constrained_variable(
    model::Optimizer,
    set_type::Type{Tuple{L,U}},
) where {L<:MOI.GreaterThan,U<:MOI.LessThan}
    return MOI.supports_add_constrained_variable(_storage_model(model), set_type)
end

function MOI.supports_add_constrained_variables(model::Optimizer, set)
    return MOI.supports_add_constrained_variables(model, typeof(set))
end

function MOI.supports_add_constrained_variables(
    model::Optimizer,
    set_type::Type{MOI.Reals},
)
    return MOI.supports_add_constrained_variables(_storage_model(model), set_type)
end

function MOI.supports_add_constrained_variables(
    model::Optimizer,
    set_type::Type{<:MOI.AbstractVectorSet},
)
    return MOI.supports_add_constrained_variables(_storage_model(model), set_type)
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(_storage_model(model), F, S)
end

MOI.is_valid(model::Optimizer, index::MOI.Index) = MOI.is_valid(model.outer_model, index)
