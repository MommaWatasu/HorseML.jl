"""
    Lasso(; alpha = 0.1, tol = 1e-4, mi = 1e+8)
Lasso Regression structure. eEach parameters are as follows:
- `alpha` : leaarning rate.
- `tol` : Allowable error.
- `mi` : Maximum number of learning.

# Example
```jldoctest regression
julia> model = Lasso()
Lasso(Float64[], 0.1, 0.0001, 100000000)

julia> fit!(model, x, t)
3-element Vector{Float64}:
   0.0
   0.5022766549841176
 154.43624186616267

julia> model(x)
20-element Vector{Float64}:
 188.64541774549468
 187.30088075704523
 181.5191726873298
 184.15748544986084
 185.72158909964372
 176.33798923891868
 185.80014722707335
 184.71565381947883
 189.20524796663838
 189.67502263476888
 189.50409373058318
 188.39535519538825
 188.481083670683
 188.88872347085172
 182.8477136378307
 188.64156231429416
 181.43996475587224
 188.9400571253936
 179.39836073711297
 185.6065850765288
```
"""
mutable struct Lasso
    w::Array{Float64, 1}
    α::Float64
    tol::Float64
    mi::Int64
    Lasso(; alpha = 0.1, tol = 1e-4, mi = 1e+8) = new(Array{Float64}(undef, 0), alpha, tol, mi)
end

sfvf(x, y) = sign(x) * max(abs(x) - y, 0)

function fit!(model::Lasso, x, t)
    function update!(x, t, w, α)
        n, d = size(x)
        w[1] = mean(t - x' * w[2:end])
        wvec = fill!(Array{Float64}(undef, d), w[1])
        for k in 1 : n
            ww = w[2:end]
            ww[k] = 0
            q = (t - wvec - x' * ww) ⋅ x[k, :]
            r = x[k, :] ⋅ x[k, :]
            w[k+1] = sfvf(q / r, α)
        end
    end
    α, tol, mi = model.α, model.tol, model.mi
    check_size(x, t)
    if ndims(x) == 1 x = x[:, :] end
    w = zeros(size(x, 1) + 1)
    e = 0.0
    for _ in 1 : mi
        eb = e
        update!(x, t, w, α)
        e = sum(abs.(w)) / length(w)
        abs(e - eb) <= tol && break
    end
    model.w = w[end:-1:1]
end

(model::Lasso)(x) = expand(x)' * model.w