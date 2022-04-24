#const LOG_2π = log(2π)
#const newaxis = [CartesianIndex()]

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

#error function
#if absolute of `x` is smaller than 2.4, we use Taylor expansion.
#other wise, we use Continued fraction expansion
function erf(x)
    absx = abs(x)
    if absx<2.4
        c=1
        a=0
        for i in 1 : 40
            a+=(x/(2i-1)*c)
            c=-c*x^2/i
        end
        return a*2/sqrt(π)
    else
        if absx>1e50
            a = 1
        else
            y = absx*sqrt(2)
            a = 0
            for i in 40:-1:1
                a=i/(y+a)
            end
            a=1-exp(-x^2)/(y+a)*sqrt(2/π)
        end
        if x<0
            return -a
        else
            return a
        end
    end
end
erfc(x) = 1-erf(x)
norm_cdf(x) = 1/2*erfc(-x/sqrt(2))

function process_parameters(dim, mean, cov)
    if dim === nothing
        if mean === nothing
            if cov === nothing
                dim = 1
            else
                cov = convert(Array{Float64}, cov)
                if ndims(cov) < 2
                    dim = 1
                else
                    dim = size(cov, 1)
                end
            end
        else
            mean = convert(Array{Float64}, mean)
            dim = length(mean)
        end
    else
        !isa(dim, Number) && throw(DimensionMismatch("dimension of random variable must be a scalar"))
    end
    
    if mean === nothing
        mean = zeros(dim)
    end
    mean = convert(Array{Float64}, mean)
    
    if cov === nothing
        cov = [1.0]
    end
    cov = convert(Array{Float64}, cov)
    
    if dim == 1
        mean = reshape(mean, 1)
        cov = reshape(cov, 1, 1)
    end
    
    if ndims(mean) != 1 || size(mean, 1) != dim
        throw(ArgumentError("array `mean` must be vector of length $dim"))
    end
    if ndims(cov) == 0
        cov = cov * Matrix{Float64}(I, dim, dim)
    elseif ndims(cov) == 1
        cov = diag(cov)
    else
        size(cov) != (dim, dim) && throw(DimensionMismatch("array `cov` must be at most two-dimensional, but ndims(cov) = $(ndims(cov))"))
    end
    return dim, mean, cov
end

function process_quantiles(x, dim)
    x = convert(Array{Float64}, x)
    
    if ndims(x) == 0
        x = [x]
    elseif ndims(x) == 1
        if dim == 1
            x = x[:, :]
        else
            x = x[newaxis, :]
        end
    end
    return x
end

function pinv_1d(v; _eps=1e-5)
    return [(abs(x)<_eps) ? 0 : 1/x for x in v]
end

function psd_pinv_decomposed_log_pdet(mat; cond=nothing, rcond=nothing)
    u, s = eigvecs(mat), eigvals(mat)
    
    if rcond !== nothing
        cond = rcond
    end
    if cond === nothing || cond == -1
        cond = 1e6 * Base.eps()
    end
    _eps = cond * maximum(abs.(s))
    
    if minimum(s) < -_eps
        throw(ArgumentError("the covariance matrix must be positive semidefinite"))
    end
    s_pinv = pinv_1d(s, _eps=_eps)
    U = u .* sqrt.(s_pinv')
    log_pdet = sum(log.(s[findall(s.>_eps)]))

    return U, log_pdet
end

function squeeze_output(out)
    if length(out) == 1
        out = Float64(out...)
    else
        out = vec(out)
    end
    return out
end

function _logpdf(x, mean, prec_U, log_det_cov)
    dim = size(x, ndims(x))
    dev = x - mean'
    tmp = (dev * prec_U).^2
    maha = sum(tmp, dims=ndims(tmp))
    maha = dropdims(maha, dims=ndims(tmp))
    return -0.5 * ((dim*LOG_2π+log_det_cov).+maha)
end

function pdf(x, mean, cov)
    dim, mean, cov = process_parameters(nothing, mean, cov)
    x = process_quantiles(x, dim)
    prec_U, log_det_cov = psd_pinv_decomposed_log_pdet(cov)
    out = exp.(_logpdf(x, mean, prec_U, log_det_cov))
    return squeeze_output(out)
end

logpdf(x, mean, cov) = log.(pdf(x, mean, cov))