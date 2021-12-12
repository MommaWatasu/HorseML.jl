function num_conbination(n::Int64, d::Int64)
    div(factorial(n+d), factorial(n)*factorial(d)) - 1
end


"""
    make_design_matrix(x, dims)
This function return the design matrix.

# Example
```jldoctest regression
julia> make_design_matrix(x, dims = 2) |> size
(20, 5)
```
"""
function make_design_matrix(x; dims = 2)
    n_features, n_datas = size(x)
    out_features = num_conbination(n_features, dims)
    y = Array{Float64}(undef, out_features, n_datas)
    current = 1
    y[current : current+n_features-1, :] = x
    index = collect(current : current+n_features-1)
    current += n_features
    push!(index, current)
    for _ in 2 : dims
        new = []
        stop = index[end]
        for feature_idx in 1 : n_features
            start = index[feature_idx]
            push!(new, current)
            next= current + stop - start
            next <= current && break
            @. y[current : next-1, :] = y[start:stop-1, :] * x[feature_idx, :]'
            current = next
        end
        push!(new, current)
        index = new
    end
    return y
end