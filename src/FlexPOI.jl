module FlexPOI

import JuMP

const MOI = JuMP.MOI
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

export Optimizer

include("optimizer/core.jl")
include("optimizer/supports.jl")
include("optimizer/runtime.jl")

include("parameters.jl")

include("attributes/model.jl")
include("attributes/constraints.jl")

include("transform.jl")

include("copy.jl")

include("updates.jl")

include("cache/core.jl")
include("cache/constraint_updates.jl")

function diff_model end

end # module FlexPOI
