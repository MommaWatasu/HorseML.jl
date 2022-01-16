using CUDA
using Base: issingletontype

@generated gpu_struct(DT, args...) = Expr(:call, nameof(DT), :(args...))

isleaf(t::T) where {T}  = issingletontype(T), !isstructtype(T), false
isleaf(t::AbstractArray{T}) where {T} = false, isprimitivetype(T), true

function array_gpu_obj(objs::AbstractArray{T}) where {T}
    println(objs)
    fields = fieldnames(T)
    return [gpu_struct(obj, [gpu(getfield(obj, f)) for f in fields]...) obj in objs]
end

function gpu(obj::T) where {T}
    is_singleton, is_leaf, is_array = isleaf(obj)
    is_leaf && return cu(obj)
    if is_array
        return array_gpu_obj(obj)
    elseif is_singleton
        return obj
    else
        fields = fieldnames(T)
        return gpu_struct(obj, [gpu(getfield(obj, f)) for f in fields]...)
    end
end

function gpu(model::NetWork)
    N = NetWork()
    n = length(model.net)
    for i in 1 : n
        D = gpu(model.net[i])
        add_layer!(N, gpu(model.net[i]))
    end
    return N
end