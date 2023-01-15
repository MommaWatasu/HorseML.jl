using HorseML
using Statistics
using Test

@testset "HorseML.jl" begin
    include("utils.jl")
    @info "Complete the test for utils"
    test_home_dir = pwd()
    include("Preprocessing.jl")
    @info "Complete the test for Preprocessing"
    cd(test_home_dir)
    include("LossFunction.jl")
    @info "Complete the test for LossFunction"
    include("./Regression.jl")
    @info "Complete the test for Regression"
    include("Classification.jl")
    @info "Complete the test for Classification"
    include("Clustering.jl")
    @info "Complete the test for Clustering"
    include("NeuralNetwork.jl")
    @info "Complete the test for NeuralNetwork"
    include("post.jl")
    @info "Complete postprocessing"
end
