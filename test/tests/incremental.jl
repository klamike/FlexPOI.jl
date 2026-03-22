@testset "Direct optimizer parameter update staging" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, x * sin(p))
    set_parameter_value(p, 2.0)
    @test isapprox(parameter_value(p), 2.0; atol = 1e-8)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(p), 2.0; atol = 1e-8)
end

@testset "Parameter updates reuse inner optimizer" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, p in Parameter(pi / 2))
    @constraint(model, c, x * sin(p) <= 1)
    @objective(model, Max, x - x * cos(p))

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL

    backend_model = backend(model)
    inner_before = backend_model.optimizer
    inner_constraint_before = backend_model.outer_to_inner_constraints[index(c)]

    set_parameter_value(p, pi / 6)
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.optimizer === inner_before
    @test backend_model.outer_to_inner_constraints[index(c)] == inner_constraint_before
end

@testset "Constraint edits stay incremental" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @constraint(model, c, x + 0 <= 1)
    @objective(model, Max, x)

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 1.0; atol = 1e-8)

    backend_model = backend(model)
    inner_before = backend_model.outer_to_inner_constraints[index(c)]

    MOI.modify(backend_model, index(c), MOI.ScalarConstantChange(-1.0))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.outer_to_inner_constraints[index(c)] == inner_before
    @test isapprox(value(x), 2.0; atol = 1e-8)

    MOI.set(backend_model, MOI.ConstraintSet(), index(c), MOI.LessThan(3.0))
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.outer_to_inner_constraints[index(c)] == inner_before
    @test isapprox(value(x), 4.0; atol = 1e-8)
end

@testset "Objective edits stay incremental" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 1)
    @objective(model, Min, x)

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 0.0; atol = 1e-8)

    backend_model = backend(model)
    inner_before = backend_model.optimizer

    MOI.modify(
        backend_model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarCoefficientChange(index(x), -1.0),
    )
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.optimizer === inner_before
    @test isapprox(value(x), 1.0; atol = 1e-8)
end

@testset "Unrelated parameter updates skip unaffected caches" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, p_obj in Parameter(pi / 2))
    @variable(model, p_con in Parameter(pi / 2))
    @constraint(model, c, x * sin(p_con) <= 5)
    @objective(model, Min, x * sin(p_obj))

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL

    backend_model = backend(model)
    objective_before = backend_model.objective_cache.current_function
    constraint_before = backend_model.scalar_constraint_caches[index(c)].current_function

    set_parameter_value(p_obj, pi / 6)
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.scalar_constraint_caches[index(c)].current_function === constraint_before
    objective_after = backend_model.objective_cache.current_function
    @test !(objective_after === objective_before)

    set_parameter_value(p_con, pi / 6)
    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test backend_model.objective_cache.current_function === objective_after
    @test !(backend_model.scalar_constraint_caches[index(c)].current_function === constraint_before)
end
