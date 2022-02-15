using DataFrames
using CSV
using HorseML.Clustering
using HorseML.Clustering: fit!

@testset "Clustering" begin
    #Load the data
    df = CSV.read("clustering.csv", DataFrame)
    K = 3
    x = Array(df)
    
    #Test for Kmeans
    model = Kmeans(K)
    @test_nowarn fit!(model, x)
    @test_nowarn model(x)
    
    #Test for GMM(Gaussian Mixture Model)
    model = GMM(K)
    @test_nowarn fit!(model, x)
    @test_nowarn model(x)
end