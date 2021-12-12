"""
    Robust()
Robust Scaler. This scaler scale data as:
``\\tilde{\\boldsymbol{x}} = \\frac{\\boldsymbol{x}-Q2}{Q3 - Q1}``

# Example
```jldoctest preprocessing
julia> scaler = Robust()
Robust(Float64[])

julia> fit!(scaler, x)
3×2 Matrix{Float64}:
 0.412913  0.739911
 0.602654  0.873014
 0.849116  0.905986

julia> transform!(scaler, x)
20×2 Matrix{Float64}:
 -0.0121192   0.041181
 -0.0260262  -0.28201
  0.649595    0.417263
  0.89357     0.764633
 -0.637774   -0.978423
  0.630675    0.0812893
  0.910919    0.3256
 -0.923097   -1.87636
 -0.0230992  -0.249283
  0.16723     0.0942497
 -0.261766   -0.742477
  0.258819   -0.041181
  0.547691    0.213832
 -0.367388   -0.392802
  0.616988    0.193438
 -0.970618   -2.10092
  0.283712    0.430314
  0.0121192   0.135449
 -1.02922    -2.64305
 -1.38159    -5.25675
```
"""
mutable struct Robust
    p::AbstractVecOrMat
    Robust() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::Robust, x; dims=1)
    if dims == 1
        scaler.p = hcat([quantile(x[:, i], [0.25, 0.5, 0.75]) for i in 1 : size(x, 2)]...)
    elseif dims == 2
        scaler.p = hcat([quantile(x[i, :], [0.25, 0.5, 0.75]) for i in 1 : size(x, 1)]...)
    end
end

rs(x, q1, q2, q3) = @. (x-q2) / (q3-q1)

function transform!(scaler::Robust, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = rs(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = rs(x[i, :], p[:, i]...)
        end
    end
    return x
end

function fit_transform!(scaler::Robust, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x; dims=dims)
end

irs(x, q1, q2, q3) = @. x*(q3-q1)+q2

function inv_transform!(scaler::Robust, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = irs(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = irs(x[i, :], p[:, i]...)
        end
    end
    return x
end