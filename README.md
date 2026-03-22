# FlexPOI

`FlexPOI.jl` rewrites JuMP models by substituting `MOI.Parameter` values directly
into the model before the problem is copied to the inner optimizer. It is similar in
spirit to ParametricOptInterface, but it works on general JuMP expressions and aims for
best-effort simplification instead of preserving a stricter parametric representation.


```julia
using JuMP
import FlexPOI
import HiGHS

const MOI = JuMP.MOI

model = direct_model(FlexPOI.Optimizer(HiGHS.Optimizer))
set_silent(model)

@variable(model, x >= 0)
@variable(model, p in MOI.Parameter(pi / 4))
@objective(model, Min, x)
@constraint(model, sin(p) * x >= 1)

optimize!(model)
set_parameter_value(p, pi / 6)
optimize!(model)
```

DiffOpt integration is also available:

```julia
using JuMP
import DiffOpt
import FlexPOI
import HiGHS

const MOI = JuMP.MOI

model = FlexPOI.quadratic_diff_model(HiGHS.Optimizer)
set_silent(model)

@variable(model, p in MOI.Parameter(2.0))
@variable(model, x)
@objective(model, Min, x^2 + sin(p) * x)

optimize!(model)

DiffOpt.set_forward_parameter(model, p, 0.1)
DiffOpt.forward_differentiate!(model)
MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())
```

The outer model can be nonlinear in parameters and still use
`FlexPOI.quadratic_diff_model` when parameter substitution leaves the inner problem
quadratic.
