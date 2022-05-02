"""
    Kmeans(K; max=300, th=1e-4)
Kmeans method.

# Parameters:
- `K`: number of class
- `max`: maximum number of repitition
- `th`: converge threshold

# Example
```jldoctest
julia> model = Kmenas(3)
Kmeans{3}(Matrix{Float64}(undef, 0, 0), 100000000, 0.0001)

julia> using HorseML.Clustering: fit!

julia> fit!(model, x)

julia> model.labels |> size
(100, 3)
```
"""
mutable struct Kmeans{K}
    μ::Matrix{Float64}
    max::Int64
    th::Float64
    labels::Vector{Int}
    Kmeans(K; max = 300, th = 1e-4) = new{K}(Array{Float64}(undef, 0, 0), max, th, [])
end
n_clusters(model::Kmeans{K}) where K = K

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
    fit!(model, x)
fit Kmeans or GMM model with data.

size of `x` is (numer of data, number of features).
"""
function fit!(model::Kmeans{K}, x) where {K}
    coverge(x, th) = @. abs(x) < th
    μ = kmeanspp(x, K)
    for _ in 1 : model.max
        μ_new = update(μ, x, K)
        if coverge.(μ_new - μ, model.th) == trues(size(μ)...)
            model.μ = μ_new
            labels = Array{Int}(undef, size(x, 1))
            r = clustering(model.μ, x, K)
            for i in 1 : size(r, 2)
                labels[findall(r[:, i].==1)] .= i
            end
            model.labels = labels
            return
        end
        μ = μ_new
    end
    @warn "Not Converged!"
    model.μ = μ
    labels = Array{Int}(undef, size(x, 1))
    r = clustering(model.μ, x, K)
    for i in 1 : size(r, 2)
        labels[findall(r[:, i].==1)] .= i
    end
    model.labels = labels
end

#initialize μ(centeroid vectors) with keams++
function kmeanspp(x, K)
    μ = Array{Float64, 2}(undef, K, size(x, 2))
    #pick first centeroid randomly
    μ[1, :] = x[rand(1:size(x, 1)), :]
    
    for i in 2 : K
        #compute the distance between each point and each centeroid
        distances = compute_distance(x, μ[1:i-1, :], size(x, 1), i-1)
        #compute the distance to the nearest centeroid
        closest_dist_sq = minimum(distances.^2, dims=2)
        #add the squares of each distances
        weights = sum(closest_dist_sq)
        
        rand_vals = rand() * weights
        #compute the culative sum of the squares of distances
        #Get the index of the data point where the cumulative sum and the value of rand_val are closest
        candidate_ids = searchsortedfirst(vec(cumsum(closest_dist_sq, dims=1)), rand_vals)
        #add centeroidx
        μ[i, :] = x[candidate_ids, :]
    end
    return μ
end

function compute_distance(x, μ, n, nc)
    dists = Array{Float64}(undef, n, nc)
    for i in 1 : nc
        dists[:, i] = sum((x.-μ[i, :]').^2, dims=2)
    end
    return dists
end