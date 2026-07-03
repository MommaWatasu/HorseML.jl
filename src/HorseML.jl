module HorseML

using LinearAlgebra
using Statistics

export Regression, Classification, Preprocessing, LossFunction, Tree, Clustering, fit!

include("utils.jl")
include("Regression/Regression.jl")
include("Classification/Classification.jl")
include("Tree/Tree.jl")
include("Clustering/Clustering.jl")
include("Preprocessing/Preprocessing.jl")
include("LossFunction/LossFunctions.jl")

end
