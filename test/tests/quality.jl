@testset "Aqua" begin
    Aqua.test_all(FlexPOI; ambiguities = false)
end
