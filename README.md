Experimental parser for parametric `JuMP.NonlinearExpr` to variable-only `JuMP.AffExpr`/`JuMP.QuadExpr` with coefficients/constant of type `Symbolics.Num`.


```julia


m = Model()
@variable m x
@variable m y
@variable m p in Parameter(1.0)
@variable m q in Parameter(2.0)

parametrizer = Parametrizer(m)
value(parametrizer, x * sin(p+cos(q)) * y) isa GenericQuadExpr{Num, VariableRef}
value(parametrizer, q * x + p) isa GenericAffExpr{Num, VariableRef}

syaff = SymbolicAffExpr(q * x + p)
syquad = SymbolicQuadExpr(x * sin(p+cos(q)) * y)
```
