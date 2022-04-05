struct Graph
    edges::Vector{Vector{Tuple{Int64, Float64}}}
    Graph(n) = new([Tuple{Int64, Float64}[] for _ in 1 : n])
end

function add_edge(G::Graph, v::Tuple{Int64, Int64, Float64})
    i, j, c = v
    push!(G.edges[i], (j, c))
    push!(G.edges[j], (i, c))
end

Base.getindex(G::Graph, i::Int) = G.edges[i]

mutable struct HDBSCANCluster
    parent::Int64
    children::Vector{Int64}
    points::Vector{Int64}
    λp::Vector{Float64}
    stability::Float64
    children_stability::Float64
    function HDBSCANCluster(noise::Bool, points::Vector{Int64})
        if noise
            return new(0, [], [], [], -1, -1)
        else
            return new(0, [], points, [], 0, 0)
        end
    end
end

Base.length(c::HDBSCANCluster) = size(c.points, 1)
join(c1::HDBSCANCluster, c2::HDBSCANCluster, id) = HDBSCANCluster(nothing, vcat(c1.points, c2.points), id, 0)
isnoise(c::HDBSCANCluster) = c.stability == -1
hasstability(c::HDBSCANCluster) = c.stability != 0
function compute_stability(c::HDBSCANCluster, λbirth)
    c.stability += sum(c.λp.-λbirth)
end

mutable struct HDBSCAN
    k::Int64
    min_cluster_size::Int64
    result::Union{Vector{Int64}, Nothing}
    function HDBSCAN(k::Int64, min_cluster_size::Int64)
        if min_cluster_size < 1
            throw(DomainError(min_cluster_size, "The `min_cluster_size` must be greater than or equal to 1"))
        end
        return new(k, min_cluster_size)
    end
end

function fit!(model::HDBSCAN, x)
    n = size(x, 1)
    #calculate core distances for each point
    core_dists = core_dist(x, model.k)
    #calculate mutual reachability distance between any two points
    mrd = mutual_reachability_distance(core_dists, x)
    #compute a minimum spanning tree by prim method
    mst = prim(mrd, n)
    #build HDBSCAN hierarchy
    hierarchy = build_hierarchy(mst, model.min_cluster_size)
    #extract the target cluster
    extract_cluster!(hierarchy)
    #generate the list of cluster assignment for each point
    result = fill(-1, n)
    for (i, j) in enumerate(hierarchy[2n-1].children)
        c = hierarchy[j]
        for k in c.points
            result[k] = i
        end
    end
    model.result = result
    return mst
end

function core_dist(points, k)
    core_dists = Array{Float64}(undef, size(points, 1))
    for i in 1 : size(points, 1)
        p = points[i:i, :]
        dists = vec(sum((@. (points - p)^2), dims=2))
        sort!(dists)
        core_dists[i] = dists[k]
    end
    return core_dists
end

function mutual_reachability_distance(core_dists, points)
    n = size(points, 1)
    graph = Graph(div(n * (n-1), 2))
    for i in 1 : n-1
        for j in i+1 : n
            c = max(core_dists[i], core_dists[j], sum((points[i, :]-points[j, :]).^2))
            add_edge(graph, (i, j, c))
        end
    end
    return graph
end

function prim(graph, n)
    minimum_spanning_tree = Array{Tuple{Float64, Int64, Int64}}(undef, n-1)
    
    marked = falses(n)
    marked_cnt = 1
    marked[1] = true
    
    h = []
    
    for (i, c) in graph[1]
        heappush!(h, (c, 1, i))
    end
    
    while marked_cnt < n
        c, i, j = popfirst!(h)
        
        marked[j] == true && continue
        minimum_spanning_tree[marked_cnt] = (c, i, j)
        marked[j] = true
        marked_cnt += 1
        
        for (k, c) in graph[j]
            marked[k] == true && continue
            heappush!(h, (c, j, k))
        end
    end
    return minimum_spanning_tree
end

function build_hierarchy(mst, min_size)
    n = length(mst) + 1
    cost = 0
    uf = UnionFind(n)
    Hierarchy = Array{HDBSCANCluster}(undef, 2n-1)
    if min_size == 1
        for i in 1 : n
            Hierarchy[i] = HDBSCANCluster(false, [i])
        end
    else
        for i in 1 : n
            Hierarchy[i] = HDBSCANCluster(true, Int64[])
        end
    end
    sort!(mst)
    
    for i in 1 : n-1
        c, j, k = mst[i]
        cost += c
        λ = 1 / cost
        #child clusters
        c1 = group(uf, j)
        c2 = group(uf, k)
        #reference to the parent cluster
        Hierarchy[c1].parent = Hierarchy[c2].parent = n+i
        nc1, nc2 = isnoise(Hierarchy[c1]), isnoise(Hierarchy[c2])
        if !(nc1 || nc2)
            #compute stability
            compute_stability(Hierarchy[c1], λ)
            compute_stability(Hierarchy[c2], λ)
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            Hierarchy[n+i] = HDBSCANCluster(false, points)
        elseif !(nc1 && nc2)
            if nc2 == true
                (c1, c2) = (c2, c1)
            end
            #record the lambda value
            append!(Hierarchy[c2].λp, fill(λ, length(Hierarchy[c1])))
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            Hierarchy[n+i] = HDBSCANCluster(false, points)
        else
            #unite the noise cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            if length(points) < min_size
                Hierarchy[n+i] = HDBSCANCluster(true, Int64[])
            else
                Hierarchy[n+i] = HDBSCANCluster(false, points)
            end
        end
    end
    return Hierarchy
end

function extract_cluster!(hierarchy::Vector{HDBSCANCluster})
    for i in 1 : length(hierarchy)-1
        if isnoise(hierarchy[i])
            c = hierarchy[i]
            push!(hierarchy[c.parent].children, i)
            hierarchy[c.parent].children_stability += c.stability
        else
            c = hierarchy[i]
            if c.stability > c.children_stability
                push!(hierarchy[c.parent].children, i)
                hierarchy[c.parent].children_stability += c.stability
            else
                append!(hierarchy[c.parent].children, c.children)
                hierarchy[c.parent].children_stability += c.children_stability
            end
        end
    end
end