@doc raw"""
    dm(x, y; reduction="mean")
Distortion Measure. This is used as an evalution function of the Kmeans model. This is the expression:
```math
DM(x, y, μ) = \sum^{N-1}_{n=0} \sum^{K-1}_{k=0} y_{nk}|| x_{n} - \mu_{k} ||^2
```
"""
function dm(x::AbstractVector, y::AbstractVector, μ::AbstractMatrix; reduction = "mean")
    J = Array{Float64}(undef, size(x, 1))
    for n in 1 : size(x, 1)
        J[n] = sum(y[n, :] .* sum(x[n, :]' .- μ, dims=2))
    end
    if reduction=="none"
        return J
    elseif reduction=="sum"
        return sum(J)
    elseif reduction=="mean"
        return mean(J)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end
function dm(x, y, μ)
    J = Array{Float64}(undef, size(x, 1))
    for n in 1 : size(x, 1)
        J[n] = sum(y[n, :] .* sum(x[n, :]' .- μ, dims=2))
    end
    return J
end
function dm(x::Number, y::Number, μ::AbstractMatrix)
    throw(DomainError("distortion measure doesn't support numbers but Arrays."))
end

function dm(y::AbstractMatrix{TY}, t::AbstractVector{TT}, μ::AbstractMatrix; reduction::String="mean") where {TY, TT}
    if length(filter(!isone, size(y))) != 1
        throw(DimensionMismatch("LossFunctions don't support for Matrix!"))
    end
    dm(vec(y), t, μ, reduction=reduction)
end
function dm(y::AbstractVector{T}, t::Number, μ::AbstractMatrix; reduction::String="mean") where {T}
    dm(y, fill(t, length(y)), μ, reduction=reduction)
end
function dm(y::AbstractMatrix{T}, t::Number, μ::AbstractMatrix; reduction::String="mean") where {T}
    if length(y) == 1
        return dm(y..., t, μ)
    elseif size(y, 1)==1 || size(y, 2)==1
        return dm(vec(y), t, μ, reduction=reduction)
    else
        throw(DimensionMismatch("LossFunctions don't support for Matrix!"))
    end
end

function gauss(x, μ, σ)
    N, D = size(x)
    c1 = 1 / (2 * π)^(D/2)
    c2 = 1 / det(σ)^0.5
    c3 = x .- μ'
    c4 = c3 * σ'
    c5 = zeros(N)
    for d in 1 : D
        c5 += c4[:, d] .* c3[:, d]
    end
    p = @. c1 * c2 * exp(-c5 / 2)
    return p
end

function nlh(x, π, μ, σ)
    N, D = size(x)
    K = length(π)
    y = Array{Float32}(undef, N, K)
    for k in 1 : K
        y[:, k] = gauss(x, μ[k, :], σ[k, :, :])
    end
    return sum(log.(sum(π' .* y, dims=2)))
end