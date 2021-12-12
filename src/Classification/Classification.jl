module Classification

using LinearAlgebra
export SVC, Logistic

include("utils.jl")
include("Logistic.jl")
include("SVC.jl")
end