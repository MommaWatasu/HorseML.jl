#This is Normal PCA(Principal COmponents Analyzer)
mutable struct PCA
    contributions::Vector{Float64}
    component_vecs::Matrix{Float64}
    PCA() = new(Vector{Float64}(undef, 0), Matrix{Float64}(undef, 0, 0))
end

@dataframe_func function fit!(pca::PCA, x::AbstractMatrix)
    #normalize
    x_ = x .- mean(x, dims=1)
    
    U, vals, vecs = svd(x_)
    pca.contributions = sort!([val / sum(vals) for val in vals], rev = true)
    perm = sortperm(vals, rev = true)
    pca.component_vecs = vecs[:, perm]
end

@dataframe_func function transform(pca::PCA, x::AbstractMatrix; n_components::Int = size(x, 2))
    n_components > size(x, 2) && throw(DimensionMismatch("n_components must be less than or equal to `size(x, 2)`"))
    x_ = x .- mean(x, dims=1)
    return x_ * pca.component_vecs[:, 1:n_components]
end

@dataframe_func function fit_transform!(pca::PCA, x::AbstractMatrix; n_components::Int = size(x, 2))
    fit!(pca, x)
    return transform(pca, x; n_components = n_components)
end

mutable struct KernelPCA
    component_vecs::Matrix{Float64}
    kernel::Symbol
    γ::Union{Nothing, Float64}
    ε::Float64
    degree::Int
    function KernelPCA(;kernel::Symbol = :gauss,  gamma::Union{Nothing, Float64} = nothing, epsilon::Float64 = 1.0, degree::Int = 3)
        if !(kernel in [:gauss, :exp, :sigmoid, :poly])
            throw(DomainError("`kernel` must be gauss, exp, sigmoid or poly"))
        end
        new(Matrix{Float64}(undef, 0, 0), kernel, gamma, epsilon, degree)
    end
end

#utils#
#return distance matrix
function pairwise(x::AbstractMatrix)
    n, m = size(x)
    r = Matrix{Float32}(undef, n, n)
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = 0
        xj = view(x, j, :)
        for i = (j + 1):n
            r[i, j] = sum((view(x, i, :) - xj).^2)
        end
    end
    return r
end

function make_kernel_matrix(K::Symbol, γ::Float64, ε::Float64, degree::Int, x)
    if K == :gauss
        return exp.(-γ*pairwise(x))
    elseif K == :exp
        return exp.(-γ*x*x' .+ ε)
    elseif K == :sigmoid
        return tanh.(γ*x*x' .+ ε)
    elseif K == :poly
        return (γ*x*x' .+ ε).^degree
    end
end
#utils#

@dataframe_func function fit!(kpca::KernelPCA, x::AbstractMatrix)
    γ = (kpca.γ　==  nothing) ? 1 / size(x, 2) : kpca.γ
    K = make_kernel_matrix(kpca.kernel, γ, kpca.ε, kpca.degree, x)
    n = size(x, 1)
    H = diagm(0 => ones(n)) - ones(n)*ones(n)'/n
    K_ = H*K
    U, vals, vecs = svd(K_)
    perm = sortperm(vals, rev = true)
    kpca.component_vecs = vecs[:, perm]
    return
end

@dataframe_func function transform(kpca::KernelPCA, x::AbstractMatrix; n_components::Int = size(x, 2))
    n_components > size(x, 2) && throw(DimensionMismatch("n_components must be less than or equal to `size(x, 2)`"))
    return x * kpca.component_vecs[1:size(x, 2), 1:n_components]
end

@dataframe_func function fit_transform!(kpca::KernelPCA, x::AbstractMatrix; n_components::Int = size(x, 2))
    fit!(kpca, x)
    return transform(kpca, x, n_components = n_components)
end