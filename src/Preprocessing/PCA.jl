#This is Normal PCA(Principal COmponents Analyzer)
mutable struct PCA
    contributions::Vector{Float64}
    component_vecs::Matrix{Float64}
    PCA_SVD() = new(Vector{Float64}(undef, 0), Matrix{Float64}(undef, 0, 0))
end

function fit!(pca::PCA, x)
    #normalize
    x_ = x .- mean(x, dims=1)
    
    U, vals, vecs = svd(x_)
    pca.contributions = sort!([val / sum(vals) for val in vals], rev = true)
    perm = sortperm(vals, rev = true)
    pca.component_vecs = vecs[:, perm]
end

function transform(pca::PCA, x; n_components::Int = size(x, 2))
    n_components > size(x, 2) && throw(DimensionMismatch("n_components must be less than or equal to `size(x, 2)`"))
    x_ = x .- mean(x, dims=1)
    return x_ * pca.component_vecs[:, 1:n_components]
end

function fit_transform!(pca::PCA, x; n_components::Int = size(x, 2))
    fit!(pca, x)
    return transform(pca, x; n_components = n_components)
end

#This is KernelPCA
@enum Kernel gaussK expK sigmoidK polyK

mutable struct KernelPCA
    component_vecs::Matrix{Float64}
    kernel::Kernel
    γ::Union{Nothing, Float64}
    ε::Float64
    degree::Int
    KernelPCA(;kernel::Kernel = gaussK,  gamma::Union{Nothing, Float64} = nothing, epsilon::Float64 = 1.0, degree::Int = 3) = new(Matrix{Float64}(undef, 0, 0), kernel, gamma, ε, degree)
end

#utils#
#return distance matrix
function pairwise(x::AbstractMatrix)
    n, m = size(x)
    r = Matrix{eltype(x)}(undef, n, n)
    z = zero(eltype(r))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            r[i, j] = r[j, i]
        end
        r[j, j] = z
        xj = view(x, j, :)
        for i = (j + 1):n
            r[i, j] = sum((view(x, i, :) - xj).^2)
        end
    end
    return r
end

function make_kernel_matrix(K::Kernel, γ::Float64, ε::Float64, degree::Int, x)
    if K == gaussK
        return exp.(-γ*pairwise(x))
    elseif K == expK
        return exp.(γ*x*x' .+ ε)
    elseif K == sigmoidK
        return tanh.(γ*x*x' .+ ε)
    else
        return (γ*x*x' .+ ε).^degree
    end
end
#utils#

function fit!(kpca::KernelPCA, x::AbstractMatrix)
    γ = (kpca.γ　==  nothing) ? 1 / size(x, 2) : kpca.γ
    K = make_kernel_matrix(kpca.kernel, γ, kpca.ε, kpca.degree, x)
    n = size(x, 1)
    H = diagm(0 => ones(n)) - ones(n)*ones(n)'/n
    K_ = H*K
    vals, vecs = eigen(K_)
    perm = sortperm(vals, rev = true)
    kpca.component_vecs = vecs[:, perm]
end

function transform(kpca::KernelPCA, x; n_components::Int = size(x, 2))
    n_components > size(x, 2) && throw(DimensionMismatch("n_components must be less than or equal to `size(x, 2)`"))
    return x * kpca.component_vecs[1:size(x, 2), 1:n_components]
end

function fit_transform!(kpca::KernelPCA, x; n_components::Int = size(x, 2))
    fit!(kpca, x)
    return transform(kpca, x, n_components = n_components)
end