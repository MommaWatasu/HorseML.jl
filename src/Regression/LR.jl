"""
    Lasso(; alpha = 0.1, tol = 1e-4, mi = 1e+8)
Lasso Regression structure.

# Parameters
- `alpha` : leaarning rate.
- `th` : Threshold to judge that it has converged.
- `max` : Maximum number of learning.

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
    w::Vector{Float64}
    α::Float64
    max::Int64
    th::Float64
    Lasso(; alpha = 0.1, max = 1e+8, th = 1e-4) = new(Array{Float64}(undef, 0), alpha, max, th)
end

soft_threshold(y, α) = sign.(y) .* max.(abs.(y) .- α, 0)

supermum_eigen(x) = maximum(sum(abs.(x), dims=1))

function fit!(model::Lasso, x, t)
    coverge(x, th) = @. abs(x) < th
    function update(w, x, t, α, rho)
        res = t - x * w
        return soft_threshold(w + (x' * res) / rho, α / rho)
    end
    x = expand(x)
    α = model.α * size(x, 1)
    w = zeros(size(x, 2))
    rho = supermum_eigen(x' * x)
    for _ in 1 : model.max
        w_new = update(w, x, t, α, rho)
        if coverge.(w_new - w, model.th) == trues(size(w)...)
            model.w = w_new
            return
        end
        w = w_new
    end
    @warn "Not Converged!"
    model.w = w
end

(model::Lasso)(x) = expand(x) * model.w