function count_unique(x)
    u = unique(x)
    c = Array{Int64}(undef, length(u))
    for i in 1 : length(u)
        c[i] = length(findall(isequal(u[i]), x))
    end
    return u, c
end

function get_branching_shold(xs)
    u = unique(xs)
    return (u[2:end]+u[1:end-1]) / 2
end

function delta_gini_index(left, right, alpha, branch_count)
    n_left, n_right = length(left), length(right)
    n_total = n_left + n_right
    _, counts = count_unique(left)
    ratio_classes = counts / n_left
    left_gain = (n_left/n_total) * (1-sum(ratio_classes.^2))
    _, counts = count_unique(right)
    ratio_classes = counts / n_right
    right_gain = (n_right/n_total) * (1-sum(ratio_classes.^2))
    return left_gain+right_gain+alpha*branch_count
end

function branch(x, t, alpha, branch_count)
    gains = []
    rules = []
    for feature_id in 1:size(x, 2)
        xs = x[:, feature_id]
        thresholds = get_branching_shold(xs)
        for th in thresholds
            left_y = t[findall(xs.<th)]
            right_y = t[findall(xs.>=th)]
            gain = delta_gini_index(left_y, right_y, alpha, branch_count)
            push!(gains, gain)
            if rules == []
                rules = [feature_id th]
            else
                rules = vcat(rules, [feature_id th])
            end
        end
    end
    best_rule = rules[argmin(gains), :]
    best_gini = minimum(gains)
    feature_id = Int(best_rule[1])
    threshold = best_rule[2]
    left = findall(x[:, feature_id] .< threshold)
    right = findall(x[:, feature_id] .>= threshold)
    return x[left, :], t[left], x[right, :], t[right], feature_id, threshold, best_gini
end


function grow(x, t, classes, gini, alpha, branch_count)
    check_size(x, t)
    uniques, counts = count_unique(t)
    counter = Dict(zip(uniques, counts))
    class_count = [(c in keys(counter)) ? counter[c] : 0 for c in classes]
    this = Dict("class_count" => class_count, "feature_id" => nothing, "threshold" => nothing, "left" => nothing, "right" => nothing)
    if length(t) == 1
        return this
    elseif length(uniques) == 1
        return this
    end
    a = true
    for i in 1:size(x, 2)
        a = a && length(unique(x[:, i])) == 1
    end
    a && return this
    left_x, left_t, right_x, right_t, feature_id, threshold, best_gini = branch(x, t, alpha, branch_count)
    if best_gini < gini
        branch_count += 1
        left = grow(left_x, left_t, classes, best_gini, alpha, branch_count)
        right = grow(right_x, right_t, classes, best_gini, alpha, branch_count)
        return Dict("class_count" => class_count, "feature_id" => feature_id, "threshold" => threshold, "left" => left, "right" => right)
    else
        return this
    end
end

"""
    DecisionTree(; alpha = 0.01)
Normal DecisionTree. `alpha` specify the complexity of the model. If it's small, it's complicated, and if it's big, it's simple.

# Example
```jldoctest classification
julia> tree = DecisionTree()
DecisionTree(0.01, Dict{Any, Any}(), Any[])

julia> fit!(tree, x, t)
Dict{String, Any} with 5 entries:
  "left"        => Dict{String, Any}("left"=>Dict{String, Union{Nothing, Vector…
  "class_count" => [8, 13, 16]
  "threshold"   => 5.7
  "right"       => Dict{String, Any}("left"=>Dict{String, Union{Nothing, Vector…
  "feature_id"  => 1

julia> println(predict(tree, x))
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
```
"""
mutable struct DecisionTree
    α::Float64
    tree::Dict
    classes::Array{Any, 1}
    DecisionTree(; alpha = 0.01) = new(alpha, Dict(), Array{Any}(undef, 0))
end

function fit!(model::DecisionTree, x, t)
    model.classes = unique(t)
    model.tree = grow(x, t, model.classes, Inf, model.α, 1)
end

function predict(model::DecisionTree, x)
    function predict_one(x, tree)
        while tree["feature_id"] != nothing
            tree = (x[tree["feature_id"]] < tree["threshold"]) ? tree["left"] : tree["right"]
        end
        return argmax(tree["class_count"])
    end
    predicts = [predict_one(x[i, :], model.tree) for i in 1:size(x, 1)]
    d = Dict(zip(collect(1:length(model.classes)), model.classes))
    return map(x->d[x], predicts)
end