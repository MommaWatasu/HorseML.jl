module HorseML

using LinearAlgebra
using Statistics

export Regression, Classification, Preprocessing, LossFunction, Tree, NeuralNetwork

include("utils.jl")
include("./Regression/Regression.jl")
include("./Classification/Classification.jl")
include("./Tree/Tree.jl")
include("./NeuralNetwork/NN.jl")
include("./Preprocessing/Preprocessing.jl")
include("./LossFunction/LossFunctions.jl")

end