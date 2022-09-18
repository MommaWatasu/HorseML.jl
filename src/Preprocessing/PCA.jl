function PCA(x; n_components::Int = size(x, 2))
    n_components > size(x, 2) && throw(DimensionMismatch("n_components must be less than or equal to `size(x, 2)`"))
    Σ = cov(x)
    vals, vecs = eigen(Σ)
    d = Dict(vals[i] => i for i in 1 : length(vals))
    sort!(vals, rev = true)
    out = Array{eltype(vecs)}(undef, size(x, 1), 0)
    for (i, val) in enumerate(vals)
        out = hcat(out, x * vecs[:, d[val]])
        i == n_components && return out
    end
end