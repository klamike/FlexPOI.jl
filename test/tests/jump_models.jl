@testset "JuMP nonlinear objective" begin
    model = Model(() -> FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, x * sin(p))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 0.0; atol = 1e-8)
    @test isapprox(value(p), 1.0; atol = 1e-8)
    set_parameter_value(p, 2.0)
    @test isapprox(parameter_value(p), 2.0; atol = 1e-8)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 0.0; atol = 1e-8)
    @test isapprox(value(p), 2.0; atol = 1e-8)
end

@testset "JuMP affine nonlinear constraint" begin
    model = Model(() -> FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, p in Parameter(pi / 2))
    @constraint(model, x * sin(p) <= 1)
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 1.0; atol = 1e-8)
    set_parameter_value(p, pi / 6)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 2.0; atol = 1e-8)
end

@testset "JuMP quadratic objective" begin
    model = Model(() -> FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x <= 1)
    @variable(model, 0 <= y <= 1)
    @variable(model, p in Parameter(pi / 2))
    @objective(model, Min, x^2 + y^2 + x * y * sin(p) - 2x - 2y)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 2 / 3; atol = 1e-6)
    @test isapprox(value(y), 2 / 3; atol = 1e-6)
    set_parameter_value(p, 0.0)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 1.0; atol = 1e-6)
    @test isapprox(value(y), 1.0; atol = 1e-6)
end

@testset "Vector nonlinear constraints simplify to supported vector forms" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    @test MOI.supports_constraint(
        optimizer,
        MOI.VectorNonlinearFunction,
        MOI.Nonnegatives,
    )
end

@testset "JuMP vector nonlinear constraint" begin
    model = Model(() -> FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x <= 2)
    @variable(model, p in Parameter(pi / 2))
    @constraint(model, [x * sin(p), 2 - x] in MOI.Nonnegatives(2))
    @objective(model, Max, x)

    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 2.0; atol = 1e-8)

    set_parameter_value(p, -pi / 2)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 0.0; atol = 1e-8)
end

@testset "Passes through unsupported scalar nonlinear expressions" begin
    model = Model(() -> FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, x^3 * sin(p))
    err = try
        optimize!(model)
        nothing
    catch caught
        caught
    end
    @test err !== nothing
    @test !occursin("Could not quadratize", sprint(showerror, err))
end

@testset "Residual nonlinear expressions pass through to nonlinear solvers" begin
    model = Model(() -> FlexPOI.Optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, x >= 0, start = 1.5)
    @variable(model, p in Parameter(1.0))
    @constraint(model, x^3 * sin(p) >= 1)
    @objective(model, Min, x^2)

    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    @test isapprox(value(x), cbrt(1 / sin(1.0)); atol = 1e-5)

    set_parameter_value(p, pi / 2)
    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    @test isapprox(value(x), 1.0; atol = 1e-5)
end

@testset "User-defined nonlinear operators are forwarded to the inner solver" begin
    f(x) = x^2 + 1

    model = direct_model(FlexPOI.Optimizer(Ipopt.Optimizer))
    set_silent(model)
    @operator(model, op_f, 1, f)
    @variable(model, x >= 0, start = 0.5)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, op_f(x + p))

    backend_model = backend(model)
    @test :op_f in MOI.get(backend_model, MOI.ListOfSupportedNonlinearOperators())

    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    @test isapprox(value(x), 0.0; atol = 1e-5)

    set_parameter_value(p, 2.0)
    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    @test isapprox(value(x), 0.0; atol = 1e-5)
end
