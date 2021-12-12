module Regression

using LinearAlgebra
using Statistics
export LinearRegression, Lasso, Ridge, make_design_matrix

include("utils.jl")
include("Base.jl")
include("BFM.jl")
include("LR.jl")
include("RR.jl")

end