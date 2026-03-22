function diffopt_symbolics_model()
    optimizer = FlexPOI.Optimizer(
        () -> DiffOpt.diff_optimizer(
            HiGHS.Optimizer;
            allow_parametric_opt_interface = false,
        ),
    )
    MOI.set(optimizer, DiffOpt.ModelConstructor(), DiffOpt.QuadraticProgram.Model)
    return direct_model(optimizer)
end

@testset "DiffOpt nonlinear objective sensitivities" begin
    model = diffopt_symbolics_model()
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @objective(model, Min, x^2 + x * sin(p))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), -sin(1.0) / 2; atol = 1e-6)

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p, 1.0)
    DiffOpt.forward_differentiate!(model)
    @test isapprox(DiffOpt.get_forward_variable(model, x), -cos(1.0) / 2; atol = 1e-5)
    @test isapprox(
        MOI.get(model, DiffOpt.ForwardObjectiveSensitivity()),
        -sin(1.0) * cos(1.0) / 2;
        atol = 1e-5,
    )

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    @test isapprox(DiffOpt.get_reverse_parameter(model, p), -cos(1.0) / 2; atol = 1e-5)
end

@testset "DiffOpt nonlinear constraint sensitivities" begin
    model = diffopt_symbolics_model()
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(1.0))
    @constraint(model, x * sin(p) == 1)
    @objective(model, Min, x^2)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(x), 1 / sin(1.0); atol = 1e-6)

    expected_dx_dp = -cos(1.0) / sin(1.0)^2

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p, 1.0)
    DiffOpt.forward_differentiate!(model)
    @test isapprox(DiffOpt.get_forward_variable(model, x), expected_dx_dp; atol = 1e-5)

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    @test isapprox(DiffOpt.get_reverse_parameter(model, p), expected_dx_dp; atol = 1e-5)
end
