using HorseML.NeuralNetwork

@testset "NeuralNetwork" begin
    #Test for utils
    @test_nowarn Dense(10=>5, relu, set_w="He")
    @test_nowarn Dense(10=>5, relu, set_w=rand)
    @test_nowarn Dense(10=>5, relu, high_accuracy=true)
    init_w = Float32.(randn((5, 10)) ./ sqrt(10))
    init_b = Float32.(fill!(similar(init_w, size(init_w, 1)), 0))
    for layer in [Dense, Denseσ]
        @test_nowarn layer(10=>5, relu, initW=init_w, initb=init_b)
        @test_nowarn layer(10=>5, relu, initW=init_w)
    end
    @test_throws ArgumentError Dense(10=>5, relu, set_w=length)
    @test_nowarn Conv((2, 2), 2=>1, relu, set_w="He")
    @test_throws ArgumentError Conv((2, 2), 2=>1, relu, set_w=rand)
    @test_nowarn Conv((2, 2), 2=>1, relu, padding=KeepSize())
    @test_nowarn Conv((2, 2), 2=>1, relu, padding=KeepSize(), high_accuracy=true)
    @test_throws ArgumentError Dropout(-0.2)
    
    data = [(rand(Float64, 10), rand(Float64)) for i in 1 : 10]
    opt = Descent()
    NN = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh))
    loss(x, y) = LossFunction.mse(NN(x), y)
    train!(NN, loss, data, opt)
    @test_nowarn @epochs 10 train!(NN, loss, data, opt)
    @test_nowarn @simple_epochs 10 train!(NN, loss, data, opt)
    @test typeof(NN[1]) <: Dense
    println(NN)
    @test_nowarn NN(rand(10, 10))
    
    #Test for activations
    ACTIVATIONS=[
        line, σ, hardσ, hardtanh, relu,
        leakyrelu, relu6, rrelu(0.01, 0.02), elu, gelu, swish, selu,
        celu, softplus, softsign, logσ, logcosh,
        mish, tanhshrink, softshrink, trelu, lisht,
        gaussian, GCU, SQU, NCU, SSU, DSU
    ]
    for f in ACTIVATIONS
        NN = NetWork(Dense(10=>1, f))
        @test_nowarn train!(NN, loss, data, opt)
        @test_nowarn NN(rand(10))
    end
    
    NN = NetWork(Denseσ(10=>1, prelu()))
    @test_nowarn train!(NN, loss, data, opt)
    @test_nowarn train!(NN, loss, data, opt)
    
    #Test for optimizers
    for opt in [Descent(), Momentum(), AdaGrad(), Adam()]
        NN = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh))
        @test_nowarn train!(NN, loss, data, opt)
        @test_nowarn NN(rand(10))
    end
    
    #Test for Conv, Pooling, Flatten, Dropout
    data = [(rand(Float32, 10, 10, 2, 1), rand(Float32)) for i in 1 : 5]
    NN = NetWork(MeanPool((2, 2)), Conv((2, 2), 2=>1, relu), MaxPool((2, 2)), Flatten(), Dropout(0.25), Dense(4=>1, tanh))
    opt = Descent()
    @test_nowarn train!(NN, loss, data, opt)
    println(NN)
    @test_nowarn NN(rand(10, 10, 2, 1))
end