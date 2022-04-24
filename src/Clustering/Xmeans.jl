struct ClusterInfo
    data::Matrix{Float64}
    label::Int
    index::Vector{Int}
    size::Int
    df::Float64
    center::Vector{Float64}
    cov::Matrix{Float64}
    function ClusterInfo(x, index, kmeans, label::Int)
        c = findall(kmeans.labels.==label)
        data = x[c, :]
        index = index[c]
        cluster_size = size(data, 1)
        features = size(data, 2)
        df = features*(features+3)/2
        centers = kmeans.μ[label, :]
        data_cov = cov(data)
        return new(data, label, index, cluster_size, df, centers, data_cov)
    end
end

function build_cluster_info(x, k_means::Kmeans; index=nothing)
    if index === nothing
        index = collect(1:size(x, 1))
    end
    labels = 1:n_clusters(k_means)
    
    return [ClusterInfo(x, index, k_means, label) for label in labels]
end

function log_likehood(c::ClusterInfo)
    return sum([logpdf(c.data[i, :], c.center, c.cov) for i in 1 : size(c.data, 1)])
end
function cluster_bic(c::ClusterInfo)
    return -2*log_likehood(c)+c.df*log(c.size)
end
    
mutable struct Xmeans
    kinit::Int
    max::Int64
    th::Float64
    labels::Vector{Int}
    centers::Vector{Vector{Float64}}
    log_likehoods::Vector{Float64}
    sizes::Vector{Int}
    Xmeans(; kinit=2, max=300, th=1e-4) = new(kinit, max, th, [], [], [], [])
end

function fit!(model::Xmeans, x)
    xmeans_clusters = ClusterInfo[]
    
    kmeans = Kmeans(model.kinit, max=model.max, th=model.th)
    fit!(kmeans, x)
    clusters = build_cluster_info(x, kmeans)
    recursively_split!(model, clusters, xmeans_clusters)
    
    labels = Array{Int}(undef, size(x, 1))
    for (i, c) in enumerate(xmeans_clusters)
        labels[c.index] .= i
    end
    model.labels = labels
    model.centers = [cluster.center for cluster in xmeans_clusters]
    model.log_likehoods = [log_likehood(cluster) for cluster in xmeans_clusters]
    model.sizes = [cluster.size for cluster in xmeans_clusters]
end

function recursively_split!(
        model::Xmeans,
        clusters::Vector{ClusterInfo},
        xmeans_clusters::Vector{ClusterInfo}
    )
    for cluster in clusters
        if cluster.size <= 3
            push!(xmeans_clusters)
            continue
        end
        kmeans = Kmeans(2, max=model.max, th=model.th)
        fit!(kmeans, cluster.data)
        c1, c2 = build_cluster_info(cluster.data, kmeans, index=cluster.index)
        
        β = norm(c1.center-c2.center) / sqrt(det(c1.cov) + det(c2.cov))
        α = 0.5 / norm_cdf(β)
        bic = -2*(cluster.size * log(α) + log_likehood(c1) + log_likehood(c2)) + 2*cluster.df*log(cluster.size)
        
        if bic < cluster_bic(cluster)
            recursively_split!(model, [c1, c2], xmeans_clusters)
        else
            push!(xmeans_clusters, cluster)
        end
    end
end