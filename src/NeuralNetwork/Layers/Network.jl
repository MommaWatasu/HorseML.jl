using Random

include("Pooling.jl")
include("Conv.jl")

"""
    NetWork(layers...)

Connect multiple layers, and build a NeuralNetwork.
NetWork also supports index.
You can also add layers later using the add_layer!() Function.

# Example
```jldoctest
julia> N = NetWork(Dense(10=>5, relu), Dense(5=>1, relu))

julia> N[1]

Dense(IO:10=>5, σ:relu)
```
"""
struct NetWork
    net::Dict{Int, Any}
    function NetWork()
        new(Dict{Int, Any}())
    end
end

function NetWork(layers...)
    N = NetWork()
    add_layer!(N, layers...)
    return N
end

function Base.getindex(N::NetWork, i::Int)
    return N.net[i]
end

function (N::NetWork)(x)
    layers = N.net
    for i in 1 : length(layers)
        x = layers[i](x)
    end
    return x
end

function Base.show(io::IO, N::NetWork)
    layers = N.net
    for i in 1 : length(layers)
        print("Layer$i : ")
        if i != length(layers)
            println(layers[i])
        else
            print(layers[i])
        end
    end
end

"""
    Dense(in=>out, σ; set_w = "Xavier", set_b = zeros, high_accuracy=false)
Crate a traditinal `Dense` layer, whose forward propagation is given by:
    σ.(muladd(W, X, b))
The input of `x` should be a Vector of length `in`, (Sorry for you can't learn using batch. I'll implement)

# Example
```jldoctest
julia> D = Dense(5=>2, relu)
Dense(IO:5=>2, σ:relu)

julia> D(rand(Float64, 5)) |> size
(2,)
```
"""
struct Dense{W, B, F}
    w::W
    b::B
    σ::F
    function Dense(W::M, σ::F) where {M<:AbstractMatrix, F}
        b = create_bias(W, size(W, 1))
        new{M, typeof(b), F}(W, b, σ)
    end
end

function Dense(io::Pair{<:Integer, <:Integer}, σ;
        initW=nothing, initb=nothing, set_w = "Xavier", set_b = zeros, high_accuracy::Bool=false)
    if initW != nothing
        if initb != nothing
            return Dense{typeof(initW), typeof(initb), typeof(σ)}(initW, initb, σ)
        else
            return Dense(initW, σ)
        end
    else
        if initb != nothing
            w = dense_w(io..., set_w)
            return Dense{typeof(w), typeof(initb), typeof(σ)}(w, initb, σ)
        end
    end
    w = dense_w(io..., set_w)
    Dense(w, σ)
end

trainable(D::Dense) = D.w, D.b

function Base.show(io::IO, D::Dense)
    o, i = size(D.w)
    σ = D.σ
    print("Dense(IO:$i => $o, σ:$σ)")
end

function (D::Dense)(X::AbstractVecOrMat)
    W, b, σ = D.w, D.b, D.σ
    σ.(muladd(W, X, b))
end

(D::Dense)(x::AbstractArray) = reshape(D(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

"""
    Denseσ(in=>out, σ; set_w = "Xavier", set_b = zeros, high_accuracy=false)
Dense layer which you can learn the parametes of the activation function. The implementation perfectly matches Dense layer.
It is assumued that the activation function will be passed as a structure, and the parameters to be learned must be in the `w` field.

# Example
```jldoctest
julia> D = Denseσ(5=>2, relu)
Denseσ(IO:5=>2, σ:relu)

julia> D(rand(Float64, 5)) |> size
(2,)
```
"""
struct Denseσ{W, B, F}
    w::W
    b::B
    σ::F
    function Denseσ(W::M, σ::F) where {M<:AbstractMatrix, F}
        b = create_bias(W, size(W, 1))
        new{M, typeof(b), F}(W, b, σ)
    end
end

function Denseσ(io::Pair{<:Integer, <:Integer}, σ;
        initW=nothing, initb=nothing, set_w = "Xavier", set_b = zeros, high_accuracy::Bool=false)
    if initW != nothing
        if initb != nothing
            return Denseσ{typeof(initW), typeof(initb), typeof(σ)}(initW, initb, σ)
        else
            return Denseσ(initW, σ)
        end
    else
        if initb != nothing
            w = dense_w(io..., set_w)
            return Denseσ{typeof(w), typeof(initb), typeof(σ)}(w, initb, σ)
        end
    end
    w = dense_w(io..., set_w)
    Denseσ(w, σ)
end

trainable(D::Denseσ) = D.w, D.b, D.σ.w

function Base.show(io::IO, D::Denseσ)
    o, i = size(D.w)
    σ = D.σ
    print("Denseσ(IO:$i => $o, σ:$σ)")
end

function (D::Denseσ)(X::AbstractVecOrMat)
    W, b, σ = D.w, D.b, D.σ
    σ.(muladd(W, X, b))
end

(D::Denseσ)(x::AbstractArray) = reshape(D(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

"""
    Flatten()
This layer change the dimentions Image to Vector.

# Example
```jldoctest
julia> F = Flatten()
Flatten(())

julia> F(rand(10, 10, 2, 5)) |> size
(1000, )
```
"""
mutable struct Flatten
    csize::Tuple
    function Flatten()
        return new(())
    end
end

function (F::Flatten)(x::Array)
    return reshape(x, :, size(x)[end])
end

"""
    Dropout(p)
This layer dropout the input data.

# Example
```jldoctest
julia> D = Dropout(0.25)
Dropout(0.25)

julia> D(rand(10))
10-element Array{Float64,1}:
 0.0
 0.3955865029078952
 0.8157710047424143
 1.0129613533211907
 0.8060508293474877
 1.1067504108970596
 0.1461289547292684
 0.0
 0.04581776023870532
 1.2794087133638332
```
"""
struct Dropout
    p::Float64
    function Dropout(p)
        if 0<=p<=1
            new(p)
        else
            throw(ArhumentError("p must be between 0 and 1!"))
        end
    end
end

function Base.show(io::IO, D::Dropout)
    print("Dropout(")
    print(string(D.p))
    print(")")
end

_dropout_kernel(y::T, p, q) where {T} = (y > p) ? T(1 / q) : T(0)

function dropout_kernel(x, p)
    y = rand!(similar(x))
    y .= _dropout_kernel.(y, p, 1-p)
    return y
end

function (D::Dropout)(x)
    y = dropout_kernel(x, D.p)
    return x .* y
end

@adjoint function (D::Dropout)(x)
    y = dropout_kernel(x, D.p)
    return x.*y, Δ -> (Δ.*y, nothing)
end

"""
     add_layer!(model, layers...)
This function add layers to model. You can add layers when you create a NeuralNetwork with `NetWork()`, and you can also use this function to add layers later.

# Example
```jldoctest
julia> N = NetWork()

julia> add_layer!(N, Dense(10=>5, relu), Dense(5=>1, relu))

julia> N
Layer1 : Dense(IO:10 => 5, σ:relu)
Layer2 : Dense(IO:5 => 1, σ:relu)
```
"""
function add_layer!(model::NetWork, obj)
    l = length(model.net)
    model.net[l+1] = obj
end

function add_layer!(model::NetWork, layers...)
    for layer in layers
        add_layer!(model, layer)
    end
end