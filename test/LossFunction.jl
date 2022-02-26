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
end