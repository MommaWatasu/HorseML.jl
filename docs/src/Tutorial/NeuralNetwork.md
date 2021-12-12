# NeuralNetwork
Let's finally build the most powerful model, NeuralNetwork!

## Basic NetWork
In anything, it's important to first learn the basics.　Let's create a networkconsisting only of [`Dense`](@ref) layer.
```
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.LossFunction

data = Matrix(dataloader("iris"))
DS = DataSplitter(150, test_size = 0.3)
train_data, test_data = DS(data)
train_x, train_t = train_data[:, 1:4], train_data[:, 5]
train_data = zip(train_x, train_t)
test_x, test_t = test_data[:, 1:4], test_data[:, 5]

model = NetWork(Dense(4=>2, relu), Dense(2=>3, sigmoid))
loss(x, y) = mse(model(x), y)
opt = Adam()
train!(model, loss, train_data, opt)
loss(test_x, test_y)
```
I wrote it all at once, but I'm doing is preparing the data, then definig and training the model.

## Advanced Model
Now that we know hot to build a network, let's build a model for image processing using the convolution layer.
```
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.LossFunction

train, test = Matrix(dataloader("MNIST"))
train = Matrix(train)
x, t = reshape(train[:, 2:end], :, 28, 28, 1, 1), train[:, 1]
test_x, test_t = reshape(test_[:, 2:end], :, 28, 28, 1, 1), test_[:, 1]
OHE = OneHotEncoder()
t = OHE(t)
data = [(x[i, :, :, :, :], t[i, :]) for i in 1 : 60000]

model = NetWork(Conv((3, 3), 1=>1, relu), MaxPool((2, 2)), Conv((2, 2), 1=>1, relu), MaxPool((2, 2)), Flatten(), Dense(36=>10, tanh))
loss(x, y) = mse(model(x), y)
opt = Adam()
@epochs 10 train!(model, loss, data, opt)
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