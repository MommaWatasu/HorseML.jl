using HorseML.Clustering
using HorseML.Clustering: fit!

@testset "Clustering" begin
    #Generate data
    N = 100
    K = 3
    t = zeros(N, 3)
    x = zeros(N, 2)
    μ = [
        -0.5 -0.5
        0.5 1.0
        1.0 -0.5
    ]
    σ = [
        0.7 0.7
        0.8 0.3
        0.3 0.8
    ]
    _π = [0.4, 0.8, 1]
    for n in 1 : N
        wk = rand()
        for k in 1 : K
            if wk < _π[k]
                t[n, k] = 1
                break
            end
        end
        for k in 1 : 2
            x[n, k] = randn() * σ[findfirst(t[n, :].==1), k] + μ[findfirst(t[n, :].==1), k]
        end
    end
    
    #Test for Kmeans
    model = Kmeans(K)
    @test_nowarn fit!(model, x)
    @test_nowarn model(x)
    
    #Test for GMM(Gaussian Mixture Model)
    model = GMM(K)
    @test_nowarn fit!(model, x)
    @test_nowarn model(x)
end