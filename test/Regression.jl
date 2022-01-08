using HorseML.Regression

@testset "Regression" begin
    mse = LossFunction.mse

    #generating data
    x = 5 .+ 25 .* rand(20)
    t = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)
    x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
    x = hcat(x, x1)

    #Linear Regression
    @testset "Linear Regression" begin
        model = LinearRegression()
        @test_throws DimensionMismatch Regression.fit!(model, x', t)
        @test_nowarn Regression.fit!(model, x, t)
        @test size(model(x)) == size(t)
    end

    #Ridge Regression
    @testset "Ridge Regression" begin
        model = Ridge()
        @test_nowarn Regression.fit!(model, x, t)
        @test size(model(x)) == size(t)
    end

    #Lasso Regression
    @testset "Lasso Regression" begin
        model = Lasso()
        @test_nowarn Regression.fit!(model, x, t)
        @test_nowarn size(model(x)) == size(t)
    end

    #Polynomial Regression
    @testset "Polynomial Regression" begin
        model = LinearRegression()
        @test_nowarn x1 = make_design_matrix(x, dims = 2)
        @test_nowarn Regression.fit!(model, x1, t)
        @test size(model(x1)) == size(t)
    end
end