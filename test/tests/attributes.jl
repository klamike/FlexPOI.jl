@testset "Automatic differentiation backend round-trips" begin
    optimizer = FlexPOI.Optimizer(Ipopt.Optimizer)
    backend = MOI.Nonlinear.SparseReverseMode()
    MOI.set(optimizer, MOI.AutomaticDifferentiationBackend(), backend)
    @test MOI.get(optimizer, MOI.AutomaticDifferentiationBackend()) === backend
end

@testset "Start attributes round-trip across parameter updates" begin
    model = direct_model(FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x <= 3)
    @variable(model, p in Parameter(pi / 4))
    @constraint(model, c, x * sin(p) >= 1)
    @objective(model, Min, x)

    backend_model = backend(model)
    MOI.set(backend_model, MOI.VariablePrimalStart(), index(x), 1.25)
    MOI.set(backend_model, MOI.ConstraintPrimalStart(), index(c), 0.5)
    MOI.set(backend_model, MOI.ConstraintDualStart(), index(c), -0.75)

    @test MOI.get(backend_model, MOI.VariablePrimalStart(), index(x)) == 1.25
    @test MOI.get(backend_model, MOI.ConstraintPrimalStart(), index(c)) == 0.5
    @test MOI.get(backend_model, MOI.ConstraintDualStart(), index(c)) == -0.75

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL

    set_parameter_value(p, pi / 6)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL

    @test MOI.get(backend_model, MOI.VariablePrimalStart(), index(x)) == 1.25
    @test MOI.get(backend_model, MOI.ConstraintPrimalStart(), index(c)) == 0.5
    @test MOI.get(backend_model, MOI.ConstraintDualStart(), index(c)) == -0.75
end
