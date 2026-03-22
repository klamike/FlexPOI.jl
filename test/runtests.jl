using Test
using JuMP
import DiffOpt
import HiGHS
import Ipopt
import FlexPOI

const MOI = JuMP.MOI

include("tests/jump_models.jl")
include("tests/incremental.jl")
include("tests/attributes.jl")
include("tests/moi.jl")
include("tests/diffopt.jl")
