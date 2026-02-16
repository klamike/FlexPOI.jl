JuMP._is_one_for_printing(::Num) = false
JuMP._is_zero_for_printing(::Num) = false
JuMP._string_round(mode, ::typeof(abs), x::Real) = JuMP._string_round(mode, x)
JuMP._sign_string(x::Num) = " + "
JuMP._complex_convert(::Type{T}, x::Num) where {T<:Real} = x
JuMP._complex_convert_type(::Num) = Num
JuMP._complex_convert_type(::Type{T}, ::Type{Num}) where {T} = Num

struct Parametrizer <: Function
    map::Dict{VariableRef,Num}
    nothrow::Bool

    function Parametrizer(model::Model)        
        map = Dict{VariableRef, Num}()
        for x in all_variables(model)
            if is_parameter(x)
                map[x], = Symbolics.@variables $(Symbol(x))
            end
        end
        new(map, false)
    end
end

function (f::Parametrizer)(vr::VariableRef)
    is_param = JuMP.is_parameter(vr)
    return is_param ? f.map[vr] : vr
end

function JuMP.value(f::Parametrizer, ex::GenericAffExpr{T,V}) where {T,V}
    ret = ex.constant
    for (var, coef) in ex.terms
        ret += coef * f(var)
    end
    return ret
end

function JuMP.value(
    f::Parametrizer,
    ex::GenericQuadExpr{CoefType,VarType},
) where {CoefType,VarType}
    ret = value(f, ex.aff)
    for (vars, coef) in ex.terms
        ret += coef * f(vars.a) * f(vars.b)
    end
    return ret
end

function JuMP._evaluate_expr(
    registry::MOI.Nonlinear.OperatorRegistry,
    f::Parametrizer,
    expr::JuMP.GenericNonlinearExpr,
)
    # The result_stack needs to be ::Real because operators like || return a
    # ::Bool. Also, some inputs may be ::Int.
    stack, result_stack = Any[expr], Any[]  # [SymbolicsPOI] was Real[]
    while !isempty(stack)
        arg = pop!(stack)
        if arg isa GenericNonlinearExpr
            push!(stack, (arg,))  # wrap in (,) to catch when we should eval it.
            for child in arg.args
                push!(stack, child)
            end
        elseif arg isa Tuple{<:GenericNonlinearExpr}
            f_expr = only(arg)
            op, nargs = f_expr.head, length(f_expr.args)
            # TODO(odow): uses private function
            result = if !MOI.Nonlinear._is_registered(registry, op, nargs)
                model = owner_model(f_expr)
                udf = MOI.get(model, MOI.UserDefinedFunction(op, nargs))
                if udf === nothing
                    return error(
                        "Unable to evaluate nonlinear operator $op because " *
                        "it was not added as an operator.",
                    )
                end
                first(udf)((pop!(result_stack) for _ in 1:nargs)...)
            elseif nargs == 1 && haskey(registry.univariate_operator_to_id, op)
                x = pop!(result_stack)
                nocast_eval_univariate_function(f, registry, op, x)
            elseif haskey(registry.multivariate_operator_to_id, op)
                args = [pop!(result_stack) for _ in 1:nargs]  # [SymbolicsPOI] was Real[]
                float64_eval_multivariate_function(registry, op, args)  # [SymbolicsPOI] was MOI.Nonlinear.eval_multivariate_function
            elseif haskey(registry.logic_operator_to_id, op)
                @assert nargs == 2
                x = pop!(result_stack)
                y = pop!(result_stack)
                MOI.Nonlinear.eval_logic_function(registry, op, x, y)
            else
                @assert haskey(registry.comparison_operator_to_id, op)
                @assert nargs == 2
                x = pop!(result_stack)
                y = pop!(result_stack)
                MOI.Nonlinear.eval_comparison_function(registry, op, x, y)
            end
            push!(result_stack, result)
        else
            push!(result_stack, JuMP._evaluate_expr(registry, f, arg))
        end
    end
    return only(result_stack)
end

function JuMP.value(f::Parametrizer, expr::GenericNonlinearExpr)
    ret = JuMP._evaluate_expr(f, expr)
    f.nothrow || ret isa GenericNonlinearExpr && error("Could not convert expression to affine/quadratic.")
    return ret
end

JuMP._evaluate_expr(f::Parametrizer, expr) = JuMP._evaluate_expr(MOI.Nonlinear.OperatorRegistry(), f, expr)

function nocast_eval_univariate_function(
    f::Parametrizer,
    registry::MOIN.OperatorRegistry,
    op::Symbol,
    x::T,
) where {T<:Union{JuMP.AbstractJuMPScalar,Real}}
    id = registry.univariate_operator_to_id[op]
    return nocast_eval_univariate_function(f, registry, id, x)
end

function nocast_eval_univariate_function(
    ::Parametrizer,
    operator::MOIN._UnivariateOperator, x::T
) where {T<:Union{JuMP.AbstractJuMPScalar,Real}}
    ret = operator.f(x)
    return ret
end

function nocast_eval_univariate_function(
    f::Parametrizer,
    registry::MOIN.OperatorRegistry,
    id::Integer,
    x::T,
) where {T<:Union{JuMP.AbstractJuMPScalar,Real}}
    if id <= registry.univariate_user_operator_start
        v, _ = MOIN._eval_univariate(id, x)
        return v
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return nocast_eval_univariate_function(f, operator, x)
end

function float64_eval_multivariate_function(
    registry::MOIN.OperatorRegistry,
    op::Symbol,
    x::V,  # [SymbolicsPOI] was AbstractVector{T}
) where {V}
    if op == :+
        return sum(x; init = 0.0)  # [SymbolicsPOI] was zero{T}
    elseif op == :-
        @assert length(x) == 2
        return x[1] - x[2]
    elseif op == :*
        return prod(x; init = 1.0)  # [SymbolicsPOI] was one{T}
    elseif op == :^
        @assert length(x) == 2
        # Use _nan_pow here to avoid throwing an error in common situations like
        # (-1.0)^1.5.
        return MOIN._nan_pow(x[1], x[2])
    elseif op == :/
        @assert length(x) == 2
        return x[1] / x[2]
    elseif op == :ifelse
        @assert length(x) == 3
        return ifelse(Bool(x[1]), x[2], x[3])
    elseif op == :atan
        @assert length(x) == 2
        return atan(x[1], x[2])
    elseif op == :min
        return minimum(x)
    elseif op == :max
        return maximum(x)
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    ret = operator.f(x)
    return ret
end

for f in (:+, :-, :*, :^, :/, :atan, :min, :max)
    op = Meta.quot(f)
    @eval begin
        function Base.$(f)(x::GenericNonlinearExpr, y::Num)
            return GenericNonlinearExpr{variable_ref_type(x)}($op, x, y)
        end
        function Base.$(f)(x::Num, y::GenericNonlinearExpr)
            return GenericNonlinearExpr{variable_ref_type(y)}($op, x, y)
        end
    end
end
