using JuMP, Symbolics
const MOIN = MOI.Nonlinear
const OC = JuMP.OrderedCollections

include("parametrizer.jl")

struct SymbolicAffExpr
    symb::GenericAffExpr{Num, VariableRef}
    inner::AffExpr
    map::Dict{Num,Float64}
    f::Parametrizer

    function SymbolicAffExpr(ex::AbstractJuMPScalar, f::Parametrizer)
        symb = value(f, ex)
        nmap = Dict(f.map[var] => parameter_value(var) for var in keys(f.map))
        inner = AffExpr(
            Symbolics.value(Symbolics.evaluate(symb.constant, nmap)),
            OC.OrderedDict(k => Symbolics.value(Symbolics.evaluate(v, nmap)) for (k,v) in symb.terms)
        )
        new(symb, inner, nmap, f)
    end
    SymbolicAffExpr(ex::AbstractJuMPScalar) = SymbolicAffExpr(ex, Parametrizer(owner_model(ex)))
end

function update_map!(syaff::SymbolicAffExpr, new_vals::Dict{VariableRef,Float64})
    for (var, val) in keys(new_vals)
        syaff.map[f.map[var]] = val
    end
end

function update_func!(syaff::SymbolicAffExpr)
    inner.constant = Symbolics.value(Symbolics.evaluate(syaff.symb.constant, syaff.map))
    for ((var, coef), inner_term) in zip(syaff.symb.terms, syaff.inner.terms)
        inner_term.coef = Symbolics.value(Symbolics.evaluate(coef, syaff.map))
    end
end

struct SymbolicQuadExpr
    symb::GenericQuadExpr{Num, VariableRef}
    inner::QuadExpr
    map::Dict{Num,Float64}
    f::Parametrizer

    function SymbolicQuadExpr(ex::AbstractJuMPScalar, f::Parametrizer)
        symb = value(f, ex)
        nmap = Dict(f.map[var] => parameter_value(var) for var in keys(f.map))
        inner = QuadExpr(
            AffExpr(
                Symbolics.value(Symbolics.evaluate(symb.aff.constant, nmap)),
                OC.OrderedDict(k => Symbolics.value(Symbolics.evaluate(v, nmap)) for (k,v) in symb.aff.terms)
            ),
            OC.OrderedDict(k => Symbolics.value(Symbolics.evaluate(v, nmap)) for (k,v) in symb.terms)
        )
        new(symb, inner, nmap, f)
    end
    SymbolicQuadExpr(ex::AbstractJuMPScalar) = SymbolicQuadExpr(ex, Parametrizer(owner_model(ex)))
end

function update_map!(syquad::SymbolicQuadExpr, new_vals::Dict{VariableRef,Float64})
    for (var, val) in keys(new_vals)
        syquad.map[f.map[var]] = val
    end
end

function update_func!(syquad::SymbolicQuadExpr)
    inner_aff = syquad.inner.aff
    symb_aff = syquad.symb.aff
    inner_aff.constant = Symbolics.value(Symbolics.evaluate(symb_aff.constant, syquad.map))
    for ((var, coef), inner_term) in zip(symb_aff.terms, inner_aff.terms)
        inner_term.coef = Symbolics.value(Symbolics.evaluate(coef, syquad.map))
    end
    for ((vars, coef), inner_term) in zip(syquad.symb.terms, syquad.inner.terms)
        inner_term.coef = Symbolics.value(Symbolics.evaluate(coef, syquad.map))
    end
end
