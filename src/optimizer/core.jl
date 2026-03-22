const _SUPPORTED_NONLINEAR_OPERATORS = let
    registry = MOI.Nonlinear.OperatorRegistry()
    ops = Set{Symbol}()
    union!(ops, keys(registry.univariate_operator_to_id))
    union!(ops, keys(registry.multivariate_operator_to_id))
    union!(ops, keys(registry.logic_operator_to_id))
    union!(ops, keys(registry.comparison_operator_to_id))
    sort!(collect(ops); by = string)
end

mutable struct _ObjectiveCache{T}
    outer_function::MOI.AbstractScalarFunction
    current_function::MOI.AbstractScalarFunction
    uses_parameters::Bool
    parameter_dependencies::Vector{MOI.VariableIndex}
end

mutable struct _ScalarConstraintCache{T}
    outer_function::MOI.AbstractScalarFunction
    outer_set::MOI.AbstractScalarSet
    inner_index::MOI.ConstraintIndex
    current_function::MOI.AbstractScalarFunction
    current_set::MOI.AbstractScalarSet
    uses_parameters::Bool
    parameter_dependencies::Vector{MOI.VariableIndex}
end

mutable struct _VectorConstraintCache
    outer_function::MOI.AbstractVectorFunction
    outer_set::MOI.AbstractVectorSet
    inner_index::MOI.ConstraintIndex
    current_function::MOI.AbstractVectorFunction
    current_set::MOI.AbstractVectorSet
    uses_parameters::Bool
    parameter_dependencies::Vector{MOI.VariableIndex}
end

mutable struct Optimizer{T,OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    outer_model::MOIU.UniversalFallback{MOIU.Model{T}}
    optimizer_factory::Any
    with_bridge_type::Any
    with_cache_type::Any
    optimizer_attributes::Dict{Any,Any}
    parameter_values::Dict{MOI.VariableIndex,T}
    pending_parameter_values::Dict{MOI.VariableIndex,T}
    parameter_constraints::Dict{
        MOI.VariableIndex,
        MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    }
    outer_to_inner_variables::Dict{MOI.VariableIndex,MOI.VariableIndex}
    outer_to_inner_constraints::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
    objective_cache::Union{Nothing,_ObjectiveCache{T}}
    scalar_constraint_caches::Dict{MOI.ConstraintIndex,_ScalarConstraintCache{T}}
    vector_constraint_caches::Dict{MOI.ConstraintIndex,_VectorConstraintCache}
    structure_dirty::Bool
    result_available::Bool
    conflict_available::Bool
    ext::Dict{Symbol,Any}
end

function _instantiate_optimizer(::Type{T}, optimizer_factory, with_bridge_type, with_cache_type) where {T}
    inner = MOI.instantiate(optimizer_factory; with_cache_type)
    if !MOI.supports_incremental_interface(inner)
        cache = MOIU.UniversalFallback(MOIU.Model{T}())
        inner = MOIU.CachingOptimizer(cache, inner)
    end
    if with_bridge_type !== nothing
        inner = MOIB.full_bridge_optimizer(inner, with_bridge_type)
    end
    return inner
end

function Optimizer{T}(
    optimizer_factory;
    with_bridge_type = T,
    with_cache_type = nothing,
) where {T}
    inner = _instantiate_optimizer(T, optimizer_factory, with_bridge_type, with_cache_type)
    outer = MOIU.UniversalFallback(MOIU.Model{T}())
    return Optimizer{T,typeof(inner)}(
        inner,
        outer,
        optimizer_factory,
        with_bridge_type,
        with_cache_type,
        Dict{Any,Any}(),
        Dict{MOI.VariableIndex,T}(),
        Dict{MOI.VariableIndex,T}(),
        Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}}(),
        Dict{MOI.VariableIndex,MOI.VariableIndex}(),
        Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
        nothing,
        Dict{MOI.ConstraintIndex,_ScalarConstraintCache{T}}(),
        Dict{MOI.ConstraintIndex,_VectorConstraintCache}(),
        true,
        false,
        false,
        Dict{Symbol,Any}(),
    )
end

Optimizer(optimizer_factory; kwargs...) = Optimizer{Float64}(optimizer_factory; kwargs...)

function _clear_parameter_state!(model::Optimizer)
    empty!(model.parameter_values)
    empty!(model.pending_parameter_values)
    empty!(model.parameter_constraints)
    return
end

function _clear_incremental_caches!(model::Optimizer)
    empty!(model.outer_to_inner_variables)
    empty!(model.outer_to_inner_constraints)
    model.objective_cache = nothing
    empty!(model.scalar_constraint_caches)
    empty!(model.vector_constraint_caches)
    return
end

function _invalidate_solution!(model::Optimizer)
    model.result_available = false
    model.conflict_available = false
    return
end

function _invalidate_structure!(model::Optimizer)
    _clear_incremental_caches!(model)
    model.structure_dirty = true
    _invalidate_solution!(model)
    return
end

function _reset_optimizer!(model::Optimizer{T}) where {T}
    model.optimizer = _instantiate_optimizer(
        T,
        model.optimizer_factory,
        model.with_bridge_type,
        model.with_cache_type,
    )
    for (attr, value) in model.optimizer_attributes
        MOI.set(model.optimizer, attr, value)
    end
    return
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.outer_model)
    _reset_optimizer!(model)
    _clear_parameter_state!(model)
    _invalidate_structure!(model)
    return
end

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.outer_model)

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    _validate_copyable_attributes(dest, src)
    MOI.empty!(dest)
    index_map = MOIU.default_copy_to(dest, src)
    return index_map
end

MOIU.final_touch(::Optimizer, _) = nothing
