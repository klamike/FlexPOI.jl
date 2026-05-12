@testset "MOI conformance tests" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    MOI.set(optimizer, MOI.Silent(), true)
    config = MOI.Test.Config(Float64)
    MOI.Test.runtests(
        optimizer,
        config;
        exclude = [
            # These are HiGHS feasibility tolerance mismatches, not wrapper errors.
            "test_nonlinear_duals",
            "test_quadratic_duplicate_terms",
            "test_quadratic_integration",
            "test_quadratic_nonhomogeneous",
        ],
        # warn_unsupported = true,
    )
end

@testset "VariableIndex-in-Parameter constraints register parameters" begin
    model = FlexPOI.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    p = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, p, MOI.Parameter(2.0))
    constraint = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, p, x)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.add_constraint(model, constraint, MOI.GreaterThan(1.0))
    objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x)],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 0.5
end
