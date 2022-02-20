using HorseML.LossFunction
using HorseML.Clustering
using HorseML.Clustering: fit!
using DataFrames
using CSV

@testset "LossFunction" begin
    sy, st = rand(), rand()
    y, t = rand(20), rand(20)
    my1, my2, my3 = fill!(Array{Float64}(undef, 1, 1), sy), y[:, :], rand(20, 2)
    t1, t2 = rand(10, 10, 2, 3), rand(10, 10, 2, 3)
    
    LOSSES = [
        mse, cee, mae, huber, logcosh_loss, poisson, hinge, smooth_hinge
    ]
    
    for f in LOSSES
        @test typeof(f(sy, st)) <: Number
        @test typeof(f(y, t)) == Float64
        @test typeof(f(y, t, reduction="sum")) == Float64
        @test size(f(y, t, reduction="none")) == (20, )
        @test_throws ArgumentError f(y, t, reduction="NaN")
        @test f(my1, st) == f(sy, st)
        @test f(my2, st) == f(y, st)
        @test_throws DimensionMismatch f(my3, st)
        @test f(my2, t) == f(y, t)
        @test_throws DimensionMismatch f(my3, t)
        @test_nowarn f(t1, t2)
    end
    
    df = CSV.read("clustering.csv", DataFrame)
    x = Array(df)
    model = Kmeans(3)
    fit!(model, x)
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
end