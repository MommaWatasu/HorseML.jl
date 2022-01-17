using CUDA
import Zygote
using Base: issingletontype
using .NeuralNetwork

const use_cuda = Ref{Union{Nothing,Bool}}(nothing)

function check_use_cuda()
  if use_cuda[] === nothing
    use_cuda[] = CUDA.functional()
    if use_cuda[] && !CUDA.has_cudnn()
      @warn "CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
    if !(use_cuda[])
      @info """The GPU function is being called but the GPU is not accessible. 
               Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
    end
  end
end
Zygote.@nograd check_use_cuda

@generated new_struct(DT, args...) = Expr(:call, nameof(DT), :(args...))

isleaf(t::T) where {T}  = issingletontype(T), !isstructtype(T), false
isleaf(t::AbstractArray{T}) where {T} = false, isprimitivetype(T), true

adapt(x) = CUDA.cu(x)
adapt(x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))
adapt(x::Zygote.OneElement) = CUDA.cu(collect(x))

cpu(x::AbstractArray) = Array(x)

for processor in (:gpu, :cpu)
    @eval begin function $(Symbol("array_$(processor)_obj"))(objs::AbstractArray{T}) where {T}
            fields = fieldnames(T)
            return new_struct(obj, [$(processor)(getfield(obj, f)) for f in fields]...)
        end
    end
    
    @eval begin
        function $(processor)(obj::T) where {T}
            is_singleton, is_leaf, is_array = isleaf(obj)
            is_leaf && return $((processor==:gpu) ? :adapt : :cpu)(obj)
            if is_array
                return array_obj(obj)
            elseif is_singleton
                return obj
            else
                fields = fieldnames(T)
                return new_struct(obj, [$(processor)(getfield(obj, f)) for f in fields]...)
            end
        end
    end
end

function gpu(model::NetWork)
    check_use_cuda()
    if use_cuda[]
        @warn "your computer doesn't have a GPU, or couldn't recognized. So a GPU isn't used."
        return model
    end
    N = NetWork()
    n = length(model.net)
    for i in 1 : n
        add_layer!(N, gpu(model.net[i]))
    end
    return N
end

function cpu(model::NetWork)
    N = NetWork()
    n = length(model.net)
    for i in 1 : n
        add_layer!(N, cpu(model.net[i]))
    end
    return N
end