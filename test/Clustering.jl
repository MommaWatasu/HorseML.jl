using DataFrames
using CSV
using HorseML.LossFunction
using HorseML.Clustering
using HorseML.Clustering: fit!

@testset "Clustering" begin
    sy, st = rand(), rand()
    y, t = rand(20), rand(20)
    my1, my2, my3 = fill!(Array{Float64}(undef, 1, 1), sy), y[:, :], rand(20, 2)
    
    df = CSV.read("clustering.csv", DataFrame)
    K = 3
    x = Array(df)
    
    #Test for Kmeans
    model = Kmeans(K)
    @test_nowarn fit!(model, x)
    
    μ = model.μ
    @test_throws DomainError dm(sy, st, μ)
    @test typeof(dm(y, t, μ)) == Float64
    @test typeof(dm(y, t, μ, reduction = "sum")) == Float64
    @test size(dm(y, t, μ, reduction = "none")) == (20, )
    @test_throws ArgumentError dm(y, t, μ, reduction="NaN")
    @test_throws DomainError dm(my1, st, μ)
    @test dm(my2, st, μ) == dm(vec(my2), st, μ)
    @test_throws DimensionMismatch dm(my3, st, μ)
    @test dm(my2, t, μ) == dm(y, t, μ)
    @test_throws DimensionMismatch dm(my3, t, μ)
    
    #Test for GMM(Gaussian Mixture Model)
    model = GMM(K)
    @test_nowarn fit!(model, x)
    
    π, μ, Σ = model.π, model.μ, model.Σ
    @test nlh(x, π, μ, Σ) ≈ 215.78213657665862
    
    #Test for DBSCAN
    data = CSV.read("clustering2.csv", DataFrame) |> Matrix
    model = DBSCAN(0.3, 3)
    @test_nowarn model(data)
    
    #Test for HEBSCAN
    model = HDBSCAN(5, 3)
    @test_nowarn mst = fit!(model, data)
    
    #Test for Xmeans
    #note that this doesn't always success
    data = CSV.read("clustering3.csv", DataFrame) |> Matrix
    model = Xmeans()
    @test_skip fit!(model, data)
end