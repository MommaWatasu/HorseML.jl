module NeuralNetwork

using LinearAlgebra
using NNlib
using Zygote
using Zygote: @adjoint

export NetWork, Dense, Denseσ, Conv, Flatten, Dropout, MaxPool, MeanPool, add_layer!, train!, @epochs, @simple_epochs, params, KeepSize

include("utils.jl")
include("activations.jl")
include("progress.jl")
include("Layers/Network.jl")
include("Optimize/train.jl")

end