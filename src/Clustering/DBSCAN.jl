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

mutable struct DBSCAN
    ε::Float64
    minpts::Int64
    label::Int64
    points::Array{Point, 1}
    DBSCAN(ε::Float64, minpts::Int64) = new(ε^2, minpts, 0, Array{Point}(undef, 0))
end

function (model::DBSCAN)(x)
    model.points = [Point(x[i, :]) for i in 1 : size(x, 1)]
    for p in model.points
        if p.not_visited
            p.not_visited = false
            neighbor_pts = range_query(model.points, p, model.ε)
            if length(neighbor_pts) < model.minpts
                # NOISE
                p.pt = false
            else
                # CORE POINT
                p.pt = true
                expand_cluster(model, p, neighbor_pts)
                model.label += 1
            end
        end
    end
    return [p.label for p in model.points]
end

function expand_cluster(model, p::Point, neighbor_pts::Array{Point, 1})
    p.label = model.label
    for q in neighbor_pts
        if q.not_visited
            q.not_visited = false
            neighbor_pts_q = range_query(model.points, q, model.ε)
            if length(neighbor_pts_q) >= model.minpts
                q.pt = true
                append!(neighbor_pts, neighbor_pts_q)
            end
        end
        if q.label == -1
            q.label = model.label
        end
    end
    
end

range_query(points::Array{Point, 1}, p::Point, ε::Float64) = filter(x -> dist(p, x) <= ε, points)