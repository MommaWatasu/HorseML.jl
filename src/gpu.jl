using CUDA
import Zygote
using Base: issingletontype
using .NeuralNetwork

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
    
    @eval begin
        function $(processor)(model::NetWork)
            N = NetWork()
            n = length(model.net)
            for i in 1 : n
                add_layer!(N, $(processor)(model.net[i]))
            end
            return N
        end
    end
end