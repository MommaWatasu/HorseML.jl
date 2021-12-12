module NeuralNetwork

using LinearAlgebra
using NNlib
using Zygote
using Zygote: @adjoint

export NetWork, Dense, Conv, Flatten, Dropout, MaxPool, add_layer!, train!, @epochs, params, KeepSize

include("utils.jl")
include("activations.jl")
include("Layers/Network.jl")
include("Optimize/train.jl")

end