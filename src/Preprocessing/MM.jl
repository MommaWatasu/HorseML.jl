"""
    MinMax()
MinMax Scaler. This scaler scale data as:
``\\tilde{\\boldsymbol{x}} = \\frac{\\boldsymbol{x}-min(\\boldsymbol{x})}{max(\\boldsymbol{x})-min(\\boldsymbol{x})}``

# Example
```jldoctest preprocessing
julia> scaler = MinMax()
MinMax(Float64[])

julia> fit!(scaler, x)
2×2 Matrix{Float64}:
  1.39855   0.954091
 -1.99789  -3.27656

julia> transform!(scaler, x)
20×2 Matrix{Float64}:
 0.597368  0.879853
 0.591301  0.826179
 0.88601   0.942311
 0.992432  1.0
 0.324455  0.710522
 0.877757  0.886514
 1.0       0.927088
 0.199996  0.561398
 0.592578  0.831614
 0.675601  0.888666
 0.488471  0.749707
 0.715552  0.866175
 0.841559  0.908526
 0.442398  0.807779
 0.871787  0.905139
 0.179268  0.524104
 0.72641   0.944478
 0.607941  0.895508
 0.153707  0.43407
 0.0       0.0
```
"""
mutable struct MinMax
    p::AbstractVecOrMat
    MinMax() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::MinMax, x; dims=1)
    scaler.p = vcat(maximum(x, dims=dims), minimum(x, dims=dims))
end 

mms(x, ma, mi) = @. (x-mi) / (ma - mi)

function transform!(scaler::MinMax, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = mms(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = mms(x[i, :], p[:, i]...)
        end
    end
    return x
end

imms(x, ma, mi) = @. x*(ma-mi)+mi

function inv_transform!(scaler::MinMax, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = imms(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = imms(x[i, :], p[:, i]...)
        end
    end
    return x
end

function fit_transform!(scaler::MinMax, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x, dims=dims)
end