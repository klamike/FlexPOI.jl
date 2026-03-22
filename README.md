# FlexPOI

An experimental package for rewriting and simplifying JuMP expressions by substitution instead of introducing additional fixed variables. Similar to POI, but supports any JuMP expression by only promising a best-effort simplification and slower parameter value updates. An integration with DiffOpt is also available.

```julia

using JuMP, FlexPOI
using HiGHS
using DiffOpt

model = FlexPOI.diff_model(HiGHS.Optimizer)
@variable(model, x >= 0)
@variable(model, θ ∈ MOI.Parameter(1.0))
@objective(model, Min, x)
@constraint(model, sin(θ) * x >= 1.0)
JuMP.optimize!(model)

DiffOpt.set_forward_parameter(model, θ, 1.0)
DiffOpt.forward_differentiate!(model)
DiffOpt.get_forward_variable(model, x)
```