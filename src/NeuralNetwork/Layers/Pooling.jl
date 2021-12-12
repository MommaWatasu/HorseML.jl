"""
    MaxPool(k::NTuple; stride = k, padding = 0)
This is a layer for max pooling with kernel size `k`.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

The default stride is the same as kernel size `k`.

# Example
```
julia> N = NetWork(Conv((2, 2), 5=>2, relu), MaxPool(2, 2))
Layer1 : Convolution(k:(2, 2), IO:5=>2, σ:relu)
Layer2 : MaxPool(k:(2, 2), stride:(2, 2) padding:(0, 0, 0, 0))

julia> x = rand(Float64, 10, 10, 5, 5) |> size
(10, 10, 5, 5)

julia> N(x) |> size
(4, 4, 5, 2)
```
"""
mutable struct MaxPool{N, M} <: NParam
    k::NTuple{N, Int}
    stride::NTuple{N, Int}
    padding::NTuple{M, Int}
end

function MaxPool(k::NTuple{N, Int}; stride = k, padding = 0) where N
    stride = expand(Val(N), stride)
    padding = expand_padding(padding, k, 1, stride)
    return MaxPool(k, stride, padding)
end

function Base.show(io::IO, MP::MaxPool)
    print("MaxPool(k:"*string(MP.k)*", stride:"*string(MP.stride)*", padding:"*string(MP.padding)*")")
end

function (mp::MaxPool)(x)
    pooldims = PoolDims(x, mp.k; padding=mp.padding, stride=mp.stride)
    return maxpool(x, pooldims)
end