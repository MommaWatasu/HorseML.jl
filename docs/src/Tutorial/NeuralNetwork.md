# NeuralNetwork
Let's finally build the most powerful model, NeuralNetwork!

## Basic NetWork
In anything, it's important to first learn the basics. Let's create a networkconsisting only of [`Dense`](@ref) layer.
```
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.LossFunction

data = Matrix(dataloader("iris"))
DS = DataSplitter(150, test_size = 0.3)
LE = LabelEncoder()
OHE = OneHotEncoder()
train_data, test_data = DS(data)
train_x, train_t = Float32.(train_data[:, 1:4]), OHE(LE(train_data[:, 5]))
train_data = [(train_x[i, :], train_t[i, :]) for i in 1 : size(train_t, 1)]

N = NetWork(Dense(4=>2, relu), Dense(2=>3, σ))
loss(x, y) = mse(N(x), y)
opt = Adam()
train!(N, loss, train_data, opt)
```
I wrote it all at once, but I'm doing is preparing the data, then definig and training the model.

## Advanced Model
Now that we know hot to build a network, let's build a model for image processing using the convolution layer.
```
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.LossFunction

train, test = Matrix.(dataloader("MNIST"))
train_x, train_t = Float32.(reshape(train[:, 2:end], :, 28, 28, 1, 1)), Float32.(train[:, 1])
test_x, test_t = reshape(test[:, 2:end], :, 28, 28, 1, 1), Float32.(test[:, 1])
train_data = [(train_x[i, :, :, :, :], train_t[i, :]) for i in 1 : 60000]
test_data = [(test_x[i, :, :, :, :], test_t[i, :]) for i in 1 : 10000]

N = NetWork(Conv((3, 3), 1=>1, relu), MaxPool((2, 2)), Conv((2, 2), 1=>1, relu), MaxPool((2, 2)), Flatten(), Dense(36=>10, tanh))
loss(x, y) = mse(N(x), y)
opt = Adam()
@epochs 10 train!(N, loss, train_data, opt)
```

## Create your layers
You can create your layer easily, like this:
```
#Layer definition
struct MyLayer
    w::AbstractArray{Float64, 2}
    b::AbstractVector{Float64}
    σ
end

#This function is called during forward propergation
function (L::MyLayer)(X::AbstractVecOrMat)
    W, b, σ = L.w, L.b, L.σ
    σ.(muladd(X, W, b))
end

#specify params
trainable(L::MyLayer) = L.w, L.b
```

## Training on GPU
Learning with large-scale data such as image data takes time if it is done with the CPU. Let's learn on the GPU (of course, this requires a GPU, so if you don't have a GPU, you'll learn as usual)!
```
using HorseML
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.LossFunction

train, test = Matrix.(dataloader("MNIST"))
train_x, train_t = Float32.(reshape(train[:, 2:end], :, 28, 28, 1, 1)) |> gpu, Float32.(train[:, 1]) |> gpu
test_x, test_t = reshape(test[:, 2:end], :, 28, 28, 1, 1), Float32.(test[:, 1])
train_data = [(train_x[i, :, :, :, :], train_t[i, :]) for i in 1 : 60000]
test_data = [(test_x[i, :, :, :, :], test_t[i, :]) for i in 1 : 10000]

N = NetWork(Conv((3, 3), 1=>1, relu), MaxPool((2, 2)), Conv((2, 2), 1=>1, relu), MaxPool((2, 2)), Flatten(), Dense(36=>10, tanh)) |> gpu
loss(x, y) = mse(N(x), y)
opt = Descent()
@epochs 10 train!(N, loss, train_data, opt)
```
