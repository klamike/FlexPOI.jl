@testset "Automatic differentiation backend round-trips" begin
    optimizer = FlexPOI.Optimizer(Ipopt.Optimizer)
    backend = MOI.Nonlinear.SparseReverseMode()
    MOI.set(optimizer, MOI.AutomaticDifferentiationBackend(), backend)
    @test MOI.get(optimizer, MOI.AutomaticDifferentiationBackend()) === backend
end
