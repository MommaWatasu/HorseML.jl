include("../Preprocessing/Encoder.jl")

"""
    SVC(; alpha=0.01, ni=1000)
Support Vector Machine Classifier.

This struct learns classifiers using One-Vs-Rest.
One-Vs-Rest generates two-class classifiers divided into one class and the other classes using Logistic Regression, adopting the most likely one among all classifiers.

Parameter `α` indicates the learning rate, and `ni` indicates the number of learnings.

# Example
```jldoctest classification
julia> model = SVC()
SVC(0.01, 1000, Logistic[])

julia> fit!(model, ct)
3-element Vector{Logistic}:
 Logistic(0.01, 1000, [0.8116709490679518 1.188329050932049; 1.7228257190036231 0.2771742809963788; -0.1519960725403138 2.1519960725403116])
 Logistic(0.01, 1000, [0.9863693439936144 1.0136306560063886; 0.8838433946106077 1.11615660538939; 1.4431044559203794 0.5568955440796174])
 Logistic(0.01, 1000, [1.262510641510418 0.7374893584895849; 0.5242383002319192 1.4757616997680822; 1.864635796779504 0.135364203220495])

julia> println(predict(model, x))
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
```
"""
mutable struct SVC
    α::Float64
    n_iter::Int64
    classifiers::Array{Logistic, 1}
    SVC(; alpha=0.01, ni=1000) = new(alpha, ni, Array{Logistic}(undef, 0))
end

function fit!(model::SVC, x, t)
    alpha, n_iter = model.α, model.n_iter
    check_size(x, t)
    c = size(t, 2)
    w = Array{Float64}(undef, 0, 0)
    classifiers = Array{Logistic}(undef, 0)
    for i in 1 : c
        classifier = Logistic(alpha = alpha, ni = n_iter)
        OHE = OneHotEncoder()
        fit!(classifier, x, OHE(t[:, i]))
        push!(classifiers, classifier)
    end
    model.classifiers = classifiers
end

function predict(model::SVC, x)
    p = Array{Float64}(undef, 0, 0)
    for i in 1 : length(model.classifiers)
        if i == 1
            p = forecast(model.classifiers[i], x)[:, 2]
        else
            p = hcat(p, forecast(model.classifiers[i], x)[:, 2])
        end
    end
    return [findfirst(p[i, :] .== maximum(p[i, :])) for i in 1:size(p, 1)]
end
