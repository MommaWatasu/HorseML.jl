using HorseML.Preprocessing
using CSV
using DataFrames

@testset "Preprocessing" begin
    
    #generating data
    x = 5 .+ 25 .* rand(20)
    t = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)
    x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
    x1 = hcat(x, x1)
    xt = x'

    #Standard Scaler
    @testset "Standard Scaler" begin
        scaler = Standard()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn Preprocessing.transform(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform(scaler, x2, dims=dim) ≈ Float32.(data)
        end
        @test_throws DimensionMismatch Preprocessing.transform(scaler, x1)
        @test_nowarn fit_transform!(scaler, x1)
    end

    #MinMaxScaler
    @testset "MinMax Scaler" begin
        scaler = MinMax()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn Preprocessing.transform(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform(scaler, x2, dims=dim) ≈ Float32.(data)
        end
        @test_throws DimensionMismatch Preprocessing.transform(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    #Robust Scaler
    @testset "Robust Scaler" begin
        scaler = Robust()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn Preprocessing.transform(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform(scaler, x2, dims=dim) ≈ Float32.(data)
        end
        @test_throws DimensionMismatch Preprocessing.transform(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    #Data Preprocessor
    @testset "Data Preprocessing" begin
        originstdin = stdin
        f = open("data/loader.txt")
        old_stdin = stdin
        redirect_stdin(f)
        @test_nowarn dataloader("MNIST")
        @test_nowarn dataloader("BostonHousing")
        @test_nowarn dataloader("iris")
        @test_throws ArgumentError dataloader("dammy.csv")
        close(f)
        redirect_stdin(old_stdin)
    end
    
    @testset "Data Splitter" begin
        DS = DataSplitter(50, test_size = 0.3)
        data = rand(50, 5)
        train, test = DS(data)
        @test size(train) == (35, 5)
        @test size(test) == (15, 5)
        train, test = DS(data', dims=2)
        @test size(train) == (5, 35)
        @test size(test) == (5, 15)
    end
    
    @testset "PCA" begin
        data = Matrix(dataloader("iris"))
        pca = PCA()
        @test_nowarn fit_transform!(pca, data[:, 1:4])
        kernels = [:gauss, :exp, :sigmoid, :poly]
        for k in kernels
            kpca = KernelPCA(kernel = k)
            @test_nowarn fit_transform!(kpca, data[:, 1:4])
        end
    end
    
    @testset "Encoders" begin
        LE = LabelEncoder()
        data = dataloader("iris")
        x, t = data[:, 1:4], data[:, 5]
        @test_nowarn LE(t)
        @test_nowarn LE(t, count=true)
        encoded_t = LE(t)
        @test LE(encoded_t, decode=true)==t
        OHE = OneHotEncoder()
        @test_nowarn OHE(encoded_t)
        t = OHE(encoded_t)
        df = CSV.read("data/preprocessing.csv", DataFrame)
        df.sex = string.(df.sex)
        OHE(df, [:sex, :birth])
    end
    
    @testset "DataBuld" begin
        @test_nowarn databuilder(x, t)
        t2 = CSV.read("data/preprocessing2.csv", DataFrame, header = false)
        @test_nowarn databuilder(x, t2)
        @test_nowarn databuilder(Matrix(x), Matrix(t2))
        @test_nowarn databuilder(Matrix(x), Matrix(t2)[:, 1])
    end
end