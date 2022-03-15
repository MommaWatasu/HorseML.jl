struct Cluster
    data::Array{Float64, 2}
    id::Int64
end

join(c1::Cluster, c2::Cluster, id::Int64) = Cluster([c1.data; c2.data], id)
Base.getindex(C::Cluster, i::Int64) = C.data[i, :]
Base.length(C::Cluster) = size(C.data, 1)

dist(d1::Vector, d2::Vector) = sum((d2 - d1) .^ 2)
dist(d1::Matrix, d2::Matrix) = [dist(d1[i, :], d2[j, :]) for j in 1 : size(d2, 1) for i in 1 : size(d1, 1)]
ward(xs::Array) = sum([dist(vec(mean(xs, dims=1)), xs[i, :])^2 for i in 1 : size(xs, 1)])
function dist(c1::Cluster, c2::Cluster, method::String)
    d = 0
    if method == "minimum"
        d = minimum(dist(c1.data, c2.data))
    elseif method == "maximum"
        d = maximum(dist(c1.data, c2.data))
    elseif method == "center"
        cv1 = mean(c1.data, dims=1)
        cv2 = mean(c2.data, dims=1)
        d = dist(cv1, cv2)
    elseif method == "mean"
        d = mean(dist(c1.data, c2.data))
    else
        d = ward(vcat(c1.data, c2.data)) - ward(c1.data) - ward(c2.data)
    end
    return d
end

struct Node
    right::Union{Cluster, Node}
    left::Union{Cluster, Node}
    clutser::Cluster
    id::Int64
end

mutable struct TreeDiagram{N}
    data::Array{Float64, 2}
    nodes::Array{Node, 1}
    tree::Union{Node, Nothing}
    count::Int64
    TreeDiagram(N, data) = new{N}(data, Array{Node}(undef, 0), nothing, 0)
end

function add_node(TD::TreeDiagram{N}, c1::Cluster, c2::Cluster) where {N}
    TD.count += 1
    c3 = join(c1, c2, TD.count+N)
    if length(c1) != 1
        for i in 1 : length(TD.nodes)
            n = TD.nodes[i]
            if n.id == c1.id
                c1 = n
                deleteat!(TD.nodes, i)
                break
            end
        end
    end
    if length(c2) != 1
        for i in 1 : length(TD.nodes)
            n = TD.nodes[i]
            if n.id == c2.id
                c2 = n
                deleteat!(TD.nodes, i)
                break
            end
        end
    end
    if TD.count == N-1
        TD.tree = Node(c1, c2, c3, 2N-1)
    else
        push!(TD.nodes, Node(c1, c2, c3, TD.count+N))
    end
    return c3
end

combinations(n) = [(i, j) for i in 1 : n for j in i+1 : n]

mutable struct Hierarchical
    tree::Union{TreeDiagram, Nothing}
    distance::String
    Hierarchical(; distance = "ward") = new(nothing, distance)
end

function (model::Hierarchical)(xs)
    N = size(xs, 1)
    clusters = [Cluster(xs[i:i, :], i) for i in 1 : N]
    TD = TreeDiagram(N, xs)
    for i in 1 : N - 1
        combination = combinations(N-i+1)
        distances = [dist(clusters[c1], clusters[c2], model.distance) for (c1, c2) in combination]
        c1, c2 = combination[argmin(distances)]
        c3 = add_node(TD, clusters[c1], clusters[c2])
        deleteat!(clusters, [c1, c2])
        push!(clusters, c3)
        println(length(clusters))
    end
end