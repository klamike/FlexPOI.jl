function _next_state!(state::Base.RefValue{UInt64})
    state[] = state[] * 6364136223846793005 + 1442695040888963407
    return state[]
end

function _rand_unit!(state::Base.RefValue{UInt64})
    return Float64(mod(_next_state!(state), UInt64(1_000_000))) / 1_000_000
end

function _rand_int!(state::Base.RefValue{UInt64}, n::Int)
    return Int(mod(_next_state!(state), UInt64(n))) + 1
end

function _rand_vector!(state::Base.RefValue{UInt64}, n::Int; scale = 1.0, shift = 0.0)
    return [shift + scale * _rand_unit!(state) for _ in 1:n]
end

function _rand_int_vector!(state::Base.RefValue{UInt64}, n::Int, upper::Int)
    return [_rand_int!(state, upper) for _ in 1:n]
end

function _randomized_problem_data(state::Base.RefValue{UInt64})
    n = 4
    m = 3
    return (
        n = n,
        m = m,
        objective_parameters = _rand_int_vector!(state, n, m),
        linear_parameters = _rand_int_vector!(state, n, m),
        constraint_parameters = _rand_int_vector!(state, n, m),
        quadratic_weights = _rand_vector!(state, n; scale = 1.0, shift = 0.5),
        linear_weights = _rand_vector!(state, n; scale = 0.3, shift = 0.1),
        coupling_weights = _rand_vector!(state, n; scale = 0.2, shift = 0.05),
        rhs = _rand_vector!(state, n; scale = 0.3, shift = 0.5),
        initial_parameters = _rand_vector!(state, m; scale = 1.4, shift = -0.7),
    )
end

function _build_randomized_model(data)
    model = direct_model(FlexPOI.Optimizer(HiGHS.Optimizer))
    set_silent(model)

    @variable(model, 0 <= x[1:data.n] <= 4)
    @variable(model, p[i = 1:data.m] in Parameter(data.initial_parameters[i]))

    @objective(
        model,
        Min,
        sum(
            (data.quadratic_weights[i] + 0.2 * sin(p[data.objective_parameters[i]])) * x[i]^2 +
            (data.linear_weights[i] + 0.1 * cos(p[data.linear_parameters[i]])) * x[i]
            for i in 1:data.n
        ),
    )

    c = @constraint(
        model,
        [
            i = 1:data.n
        ],
        x[i] + data.coupling_weights[i] * sin(p[data.constraint_parameters[i]]) * x[mod1(i + 1, data.n)] >=
        data.rhs[i],
    )
    return model, x, p, c
end

function _set_parameter_values!(parameters, values)
    for (parameter, value) in zip(parameters, values)
        set_parameter_value(parameter, value)
    end
    return
end

@testset "Randomized incremental updates match forced rebuilds" begin
    state = Ref(UInt64(0x12345678))
    for _ in 1:8
        data = _randomized_problem_data(state)
        incremental_model, x_inc, p_inc, c_inc = _build_randomized_model(data)
        rebuild_model, x_rebuild, p_rebuild, c_rebuild = _build_randomized_model(data)

        for _ in 1:6
            values = _rand_vector!(state, data.m; scale = 1.6, shift = -0.8)
            _set_parameter_values!(p_inc, values)
            _set_parameter_values!(p_rebuild, values)

            optimize!(incremental_model)
            @test termination_status(incremental_model) == MOI.OPTIMAL

            FlexPOI._invalidate_structure!(backend(rebuild_model))
            optimize!(rebuild_model)
            @test termination_status(rebuild_model) == MOI.OPTIMAL

            @test isapprox(
                objective_value(incremental_model),
                objective_value(rebuild_model);
                atol = 1e-7,
                rtol = 1e-7,
            )
            for (x_i, x_j) in zip(x_inc, x_rebuild)
                @test isapprox(value(x_i), value(x_j); atol = 1e-7, rtol = 1e-7)
            end

            incremental_backend = backend(incremental_model)
            rebuild_backend = backend(rebuild_model)
            @test typeof(incremental_backend.objective_cache.current_function) ==
                  typeof(rebuild_backend.objective_cache.current_function)
            for (ci_inc, ci_rebuild) in zip(index.(c_inc), index.(c_rebuild))
                @test typeof(
                    incremental_backend.scalar_constraint_caches[ci_inc].current_function,
                ) == typeof(
                    rebuild_backend.scalar_constraint_caches[ci_rebuild].current_function,
                )
                @test typeof(
                    incremental_backend.scalar_constraint_caches[ci_inc].current_set,
                ) == typeof(
                    rebuild_backend.scalar_constraint_caches[ci_rebuild].current_set,
                )
            end
        end
    end
end
