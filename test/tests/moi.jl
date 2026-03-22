@testset "MOI conformance tests" begin
    optimizer = FlexPOI.Optimizer(HiGHS.Optimizer)
    MOI.set(optimizer, MOI.Silent(), true)
    config = MOI.Test.Config(
        Float64;
        exclude = Any[
            MOI.ConstraintDual,
            MOI.VariablePrimalStart,
            MOI.ConstraintPrimalStart,
            MOI.ConstraintDualStart,
        ],
    )
    MOI.Test.runtests(
        optimizer,
        config;
        exclude = [
            "test_multiobjective_vector_nonlinear_modify",
            "test_quadratic_duplicate_terms",
            "test_quadratic_integration",
            "test_quadratic_nonhomogeneous",
        ],
        # warn_unsupported = true,
    )
end
