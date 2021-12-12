using HorseML.LossFunction

@testset "LossFunction" begin
    sy, st = rand(), rand()
    y, t = rand(20), rand(20)
    my1, my2, my3 = fill!(Array{Float64}(undef, 1, 1), sy), rand(20, 1), rand(20, 2)
    
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
        @test f(my2, st) == f(vec(my2), st)
        @test_throws DimensionMismatch f(my3, st)
    end
end