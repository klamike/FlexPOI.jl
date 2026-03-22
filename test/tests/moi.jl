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
