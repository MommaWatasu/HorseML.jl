"""
    Conv(kernel, in=>out, σ; stride = 1, pading = 0, set_w = "Xavier")
This is the traditional convolution layer. `kernel` is a tuple of integers that specifies the kernel size, it must have one or two elements. And, `in` and `out` specifies number of input and out channels.

The input data must have a dimensions WHCB(weight, width, channel, batch). If you want to use a data which has a dimentions WHC, you must be add a dimentioins of B.

`stride` and `padding` are single integers or tuple(stride is tuple of 2 elements, padding is tuple of 2 elements), and if you specifies `KeepSize` to padding, we adjust sizes of input and return a matrix which has the same sizes.
`set_w` is `Xavier` or `He`, it decide a method to create a first parameter. This parameter is the same as `Dense()`.

# Example
```
julia> C = Conv((2, 2), 2=>2, relu)
Convolution(k:(2, 2), IO:2 => 2, σ:relu)

julia> C(rand(10, 10, 2, 5)) |> size
(9, 9, 2, 5)
```

!!! warning
    When you specidies `same` to `padding`, in some cases, it will be returned one size smaller.
    Because of its expression.
```
julia> C = Conv((2, 2), 2=>2, relu, padding = KeepSize)
Convolution(k:(2, 2), IO:2 => 2, σ:relu

julia> C(rand(10, 10, 2, 5)) |> size
(9, 9, 2, 5)
```
"""
struct Conv{N, M, K, I, O} <: Param
    σ
    weight
    bias
    stride::NTuple{N, Int}
    padding::NTuple{M, Int}
    dilation::NTuple{N, Int}
end

function Conv(weight::AbstractArray{T, N}, σ, k::Tuple, io::Pair{<:Integer, <:Integer}; stride = 1, padding = 0, dilation=1) where {N, T}
    stride = expand(Val(N-2), stride)
    dilation = expand(Val(N-2), dilation)
    padding = expand_padding(padding, size(weight)[1:N-2], dilation, stride)
    bias = create_bias(weight, size(weight, N))
    return Conv{length(stride), length(padding), k, io...}(σ, weight, bias, stride, padding, dilation)
end

function Conv(k::NTuple{N, Int}, io::Pair{<:Integer, <:Integer}, σ; stride = 1, padding = 0, dilation=1, set_w = "Xavier") where {N}
    weight = conv_w(k, io, set_w)
    return Conv(weight, σ, k, io, stride = stride, padding = padding, dilation = dilation)
end

function Base.show(io::IO, C::Conv{N, M, K, I, O}) where {N, M, K, I, O}
    σ = C.σ
    print("Convolution(k:$K, IO:$I => $O, σ:$σ)")
end

function (C::Conv)(x)
    σ, b, w = C.σ, C.bias, C.weight
    dim = DenseConvDims(x, w, stride = C.stride, padding = C.padding, dilation = C.dilation)
    return σ.(conv(x, w, dim).+b)
end

trainable(C::Conv) = C.weight, C.bias