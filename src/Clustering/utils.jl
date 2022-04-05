# each data point
mutable struct Point
    data::Array
    label::Int64
    not_visited::Bool
    pt::Union{Nothing, Bool}
    Point(data::Array) = new(data, -1, true, nothing)
end
# calculate distance between x1 and x2
dist(p1::Point, p2::Point) = sum((p2.data - p1.data) .^ 2)

heappush!(h, v) = insert!(h, searchsortedfirst(h, v), v)

mutable struct UnionFind{T <: Integer}
    parent:: Vector{T}  # parent[root] is the negative of the size
    label::Dict{Int, Int}
    cnt::Int64

    function UnionFind{T}(nodes::T) where T<:Integer
        if nodes <= 0
            throw(ArgumentError("invalid argument for nodes: $nodes"))
        end

        parent = -ones(T, nodes)
        label = Dict([(i, i) for i in 1 : nodes])
        new{T}(parent, label, nodes)
    end
end

UnionFind(nodes::Integer) = UnionFind{typeof(nodes)}(nodes)
group(uf::UnionFind, x)::Int = uf.label[root(uf, x)]
members(uf::UnionFind, x::Int) = collect(keys(filter(n->n.second == x, uf.label)))

function root(uf::UnionFind{T}, x::T)::T where T<:Integer
    if uf.parent[x] < 0
        return x
    else
        # uf.parent[x] = root{T}(uf, uf.parent[x])
        # return uf.parent[x]
        return uf.parent[x] = root(uf, uf.parent[x])
    end
end

function issame(uf::UnionFind{T}, x::T, y::T)::Bool where T<:Integer
    return root(uf, x) == root(uf, y)
end

function Base.size(uf::UnionFind{T}, x::T)::T where T<:Integer
    return -uf.parent[root(uf, x)]
end

function unite!(uf::UnionFind{T}, x::T, y::T)::Bool where T<:Integer
    x = root(uf, x)
    y = root(uf, y)
    if x == y
        return false
    end
    if uf.parent[x] > uf.parent[y]
        x, y = y, x
    end
    # unite smaller tree(y) to bigger one(x)
    uf.parent[x] += uf.parent[y]
    uf.parent[y] = x
    uf.cnt += 1
    uf.label[y] = uf.cnt
    for i in members(uf, group(uf, x))
        uf.label[i] = uf.cnt
    end
    return true
end