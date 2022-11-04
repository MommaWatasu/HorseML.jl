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