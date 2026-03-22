function diffopt_symbolics_model()
    return FlexPOI.quadratic_diff_model(HiGHS.Optimizer)
end

function nonlinear_diffopt_symbolics_model()
    return FlexPOI.nonlinear_diff_model(Ipopt.Optimizer)
end

function _reset_parameters_and_optimize!(parameters, values, model)
    for (parameter, value) in zip(parameters, values)
        set_parameter_value(parameter, value)
    end
    optimize!(model)
    @test is_solved_and_feasible(model)
    return
end

@testset "DiffOpt model helper wrappers set constructors" begin
    generic_model = FlexPOI.diff_model(HiGHS.Optimizer)
    @test backend(generic_model) isa FlexPOI.Optimizer
    @test isnothing(MOI.get(backend(generic_model), DiffOpt.ModelConstructor()))

    quadratic_model = FlexPOI.quadratic_diff_model(HiGHS.Optimizer)
    @test backend(quadratic_model) isa FlexPOI.Optimizer
    @test MOI.get(backend(quadratic_model), DiffOpt.ModelConstructor()) ===
          DiffOpt.QuadraticProgram.Model

    conic_model = FlexPOI.conic_diff_model(HiGHS.Optimizer)
    @test backend(conic_model) isa FlexPOI.Optimizer
    @test MOI.get(backend(conic_model), DiffOpt.ModelConstructor()) ===
          DiffOpt.ConicProgram.Model

    nonlinear_model = FlexPOI.nonlinear_diff_model(Ipopt.Optimizer)
    @test backend(nonlinear_model) isa FlexPOI.Optimizer
    @test MOI.get(backend(nonlinear_model), DiffOpt.ModelConstructor()) ===
          DiffOpt.NonLinearProgram.Model
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

@testset "DiffOpt forward objective sensitivity matches finite differences" begin
    model = nonlinear_diffopt_symbolics_model()
    set_silent(model)

    p_val = 3.0
    @variable(model, p in Parameter(p_val))
    @variable(model, x >= 1)
    @objective(model, Min, p^2 * x^2)

    optimize!(model)
    @test is_solved_and_feasible(model)

    Δp = 0.1
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p, Δp)
    DiffOpt.forward_differentiate!(model)

    df_dp = MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())
    @test isapprox(df_dp, 2 * p_val * Δp; atol = 1e-7)

    ε = 1e-6
    set_parameter_value(p, p_val + ε)
    optimize!(model)
    df_dp_fdpos = Δp * objective_value(model)

    set_parameter_value(p, p_val - ε)
    optimize!(model)
    df_dp_fdneg = Δp * objective_value(model)

    set_parameter_value(p, p_val)
    optimize!(model)

    df_dp_fd = (df_dp_fdpos - df_dp_fdneg) / (2 * ε)
    @test isapprox(df_dp, df_dp_fd; atol = 1e-4)
end

@testset "DiffOpt multi-parameter nonlinear forward sensitivities match finite differences" begin
    model = nonlinear_diffopt_symbolics_model()
    set_silent(model)

    p1_val = 1.7
    p2_val = -0.6
    @variable(model, p1 in Parameter(p1_val))
    @variable(model, p2 in Parameter(p2_val))
    @variable(model, x >= 0, start = 0.25)
    @objective(model, Min, p1^2 * x^2 + sin(p2) * x)

    optimize!(model)
    @test is_solved_and_feasible(model)

    base_parameters = [p1_val, p2_val]
    Δ = [0.2, -0.15]
    ε = 1e-6

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p1, Δ[1])
    DiffOpt.set_forward_parameter(model, p2, Δ[2])
    DiffOpt.forward_differentiate!(model)

    df_forward = MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())

    _reset_parameters_and_optimize!(
        [p1, p2],
        base_parameters .+ ε .* Δ,
        model,
    )
    f_pos = objective_value(model)

    _reset_parameters_and_optimize!(
        [p1, p2],
        base_parameters .- ε .* Δ,
        model,
    )
    f_neg = objective_value(model)

    df_forward_fd = (f_pos - f_neg) / (2 * ε)

    @test isapprox(df_forward, df_forward_fd; atol = 1e-4)

    _reset_parameters_and_optimize!([p1, p2], base_parameters, model)
end

@testset "DiffOpt multi-parameter reverse variable sensitivities match finite differences" begin
    model = diffopt_symbolics_model()
    set_silent(model)

    p1_val = 0.7
    p2_val = -0.2
    @variable(model, p1 in Parameter(p1_val))
    @variable(model, p2 in Parameter(p2_val))
    @variable(model, x)
    @objective(model, Min, x^2 + x * sin(p1) + x * cos(p2))

    optimize!(model)
    @test is_solved_and_feasible(model)

    base_parameters = [p1_val, p2_val]
    Δ = [0.25, -0.1]
    ε = 1e-6

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p1, Δ[1])
    DiffOpt.set_forward_parameter(model, p2, Δ[2])
    DiffOpt.forward_differentiate!(model)

    df_forward = MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())

    _reset_parameters_and_optimize!(
        [p1, p2],
        base_parameters .+ ε .* Δ,
        model,
    )
    f_pos = objective_value(model)

    _reset_parameters_and_optimize!(
        [p1, p2],
        base_parameters .- ε .* Δ,
        model,
    )
    f_neg = objective_value(model)

    df_forward_fd = (f_pos - f_neg) / (2 * ε)
    @test isapprox(df_forward, df_forward_fd; atol = 1e-4)

    _reset_parameters_and_optimize!([p1, p2], base_parameters, model)

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dp1_reverse = DiffOpt.get_reverse_parameter(model, p1)
    dp2_reverse = DiffOpt.get_reverse_parameter(model, p2)

    _reset_parameters_and_optimize!([p1, p2], [p1_val + ε, p2_val], model)
    x_p1_pos = value(x)
    _reset_parameters_and_optimize!([p1, p2], [p1_val - ε, p2_val], model)
    x_p1_neg = value(x)

    _reset_parameters_and_optimize!([p1, p2], [p1_val, p2_val + ε], model)
    x_p2_pos = value(x)
    _reset_parameters_and_optimize!([p1, p2], [p1_val, p2_val - ε], model)
    x_p2_neg = value(x)

    dp1_fd = (x_p1_pos - x_p1_neg) / (2 * ε)
    dp2_fd = (x_p2_pos - x_p2_neg) / (2 * ε)

    @test isapprox(dp1_reverse, dp1_fd; atol = 1e-4)
    @test isapprox(dp2_reverse, dp2_fd; atol = 1e-4)

    _reset_parameters_and_optimize!([p1, p2], base_parameters, model)
end
