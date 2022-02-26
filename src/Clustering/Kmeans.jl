"""
    Kmeans(K; max=1e+8, th=1e-4)
Kmeans method.

# Parameters:
- K: number of class
- max: maximum number of repitition
- th: converge threshold

# Example
```jldoctest
julia> model = Kmenas(3)
Kmeans{3}(Matrix{Float64}(undef, 0, 0), 100000000, 0.0001)

julia> using HorseML.Clustering: fit!

julia> fit!(model, x)

julia> model(x) |> size
(100, 3)
```
"""
mutable struct Kmeans{K}
    μ::Matrix{Float64}
    max::Int64
    th::Float64
    Kmeans(K; max = 1e+8, th = 1e-4) = new{K}(Array{Float64}(undef, 0, 0), max, th)
end

function clustering(μ, x, K)
    r = zeros(size(x, 1), K)
    for n in 1 : size(x, 1)
        diff = Matrix(undef, size(μ)...)
        for k in 1 : K
            diff[k, :] = x[n, :] - μ[k, :]
        end
        distance = @. abs(diff)
        r[n, argmin(vec(sum(distance, dims=2)))] = 1
    end
    return r
end

function update(μ, x, K)
    r = clustering(μ, x, K)
    μ = similar(μ)
    for k in 1 : K
        μ[k, :] = sum(x .* r[:, k], dims=1) / sum(r[:, k])
    end
    return μ
end

"""
    fit!(model::Kmeans, x, K)
x is Number of data × Number of feature
"""
function fit!(model::Kmeans{K}, x) where {K}
    coverge(x, th) = @. abs(x) < th
    ma = maximum(x, dims=1)
    mi = minimum(x, dims=1)
    interval = (ma-mi) / (K-1)
    μ = Array{Float64, 2}(undef, K, size(x, 2))
    for i in 1 : K-1
        μ[i, :] = mi
        mi += interval
    end
    for _ in 1 : model.max
        μ_new = update(μ, x, K)
        if coverge.(μ_new - μ, model.th) == trues(size(μ)...)
            model.μ = μ_new
            return
        end
        μ = μ_new
    end
    @warn "Not Converged!"
    model.μ = μ
end

(model::Kmeans{K})(x) where{K} = clustering(model.μ, x, K)