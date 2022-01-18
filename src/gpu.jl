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
new_struct(t::T, args...) where {T<:Tuple} = ntuple(i->args[i], length(args))

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
                return $(Symbol("array_$(processor)_obj"))(obj)
            elseif is_singleton
                return obj
            else
                fields = fieldnames(T)
                return new_struct(obj, [$(processor)(getfield(obj, f)) for f in fields]...)
            end
        end
    end
end

"""
    gpu(model)
Transform the model so that it can be trained on the GPU. When called in an environment without a GPU, it does nothing and returns the original model.
!!! note
    This function is included in the HorseML module and can only be used with `using HorseML`.

# Example
```jldoctest gpu
julia> model = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh)) |> gpu
Layer1 : Dense(IO:10 => 5, σ:relu)
Layer2 : Dense(IO:5 => 1, σ:tanh)

julia> model[1].w |> typeof
CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```
"""
function gpu(model::NetWork)
    check_use_cuda()
    if !use_cuda[]
        @warn "your computer doesn't have a GPU, or couldn't recognized. So a GPU isn't used."
        return model
    end
    N = NetWork()
    n = length(model.net)
    for i in 1 : n
        add_layer!(N, gpu(model[i]))
    end
    return N
end

"""
    cpu(model)
Put the model trained on the GPU back on the CPU.
!!! note
    This function is included in the HorseML module and can only be used with `using HorseML`.

# Example
```jldoctest
julia> model_on_cpu = model |> cpu #This model is made with description of gpu function
Layer1 : Dense(IO:10 => 5, σ:relu)
Layer2 : Dense(IO:5 => 1, σ:tanh)

julia> model_on_cpu[1].w |> typeof
Matrix{Float32} (alias for Array{Float32, 2})
```
"""
function cpu(model::NetWork)
    N = NetWork()
    n = length(model.net)
    for i in 1 : n
        add_layer!(N, cpu(model[i]))
    end
    return N
end