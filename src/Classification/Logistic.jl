"""
    Logistic(; alpha = 0.01, ni = 1000)
Logistic Regression classifier. This model learns classifiers using multi class softmax.
# Parameters
- `α`: learning rate
- `ni`: maximum number of repitition

# Example
```jldoctest classification
julia> model = Logistic(alpha = 0.1)
Logistic(0.1, 1000, Matrix{Float64}(undef, 0, 0))

julia> fit!(model, x, ct)
3×3 Matrix{Float64}:
  1.80736  1.64037  -0.447735
 -1.27053  1.70026   2.57027
  4.84966 -0.473835 -1.37582

julia> println(predict(model, x))
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
```
"""
mutable struct Logistic
    α::Float64
    n_iter::Int64
    w::Array{Float64, 2}
    Logistic(; alpha = 0.01, ni=1000) = new(alpha, ni, Array{Float64}(undef, 0, 0))
end

"""
    fit!(model, x, t)
fit the model with the data.

# Parameters
- `model`: Logistic or SVC structure
- `x`: training data whose size is (number of data, number of classes)
- `t`: training data whose size is (number of data, number of classes) and encoded
"""
function fit!(model::Logistic, x, t)
    alpha, n_iter = model.α, model.n_iter
    check_size(x, t)
    x = hcat(ones(size(x, 1), 1), x)
    w = ones(size(x, 2), size(t, 2))
    Threads.@threads for n in 1 : n_iter
        w -= alpha * ceed(x, w, t)
    end
    model.w = w
end

function forecast(model::Logistic, x)
    w = model.w
    x = hcat(ones(size(x, 1), 1), x)
    return softmax(x*w)
end

function (model::Logistic)(x)
    w = model.w
    x = hcat(ones(size(x, 1), 1), x)
    s = softmax(x * w)
    return [findfirst(s[i, :] .== maximum(s[i, :])) for i in 1:size(s, 1)]
end