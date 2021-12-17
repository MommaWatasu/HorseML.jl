function dense_w(in_size, out_size, m; high_accuracy::Bool=false)
    if m == "Xavier"
        w = randn((out_size, in_size)) ./ sqrt(in_size)
    elseif m == "He"
        w = randn((out_size, in_size)) ./ sqrt(in_size) .* sqrt(2)
    else
        try
            w = m(out_size, in_size)
        catch
            throw(ArgumentError("`set_w` must be `Xavier`, `He` or a function to create weight."))
        end
    end
    if !high_accuracy
        return Float32.(w)
    else
        return w
    end
end

function conv_w(K, io, m; high_accuracy::Bool=false)
    if m == "Xavier"
        w = randn(K..., io...) ./ sqrt(io[1])
    elseif m == "He"
        w = randn(K..., io...) ./ sqrt(io[1]) .* sqrt(2)
    else
        throw(ArgumentError("Custom initialization functions aren't yet aupported!"))
    end
    if !high_accuracy
        return Float32.(w)
    else
        return w
    end
end

create_bias(w, dims...) = fill!(similar(w, dims...), 0)

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

struct KeepSize end

expand_padding(padding, k::NTuple{N, T}, dilation, stride) where {N, T} = expand(2N, padding)
function expand_padding(::KeepSize, k::NTuple{N, T}, dilation, stride) where {N, T}
    k_eff = @. k + (k - 1) * (dilation - 1)
    pad_amt = @. k_eff - 1
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end

function params(model)
    pa = Array{Any}(undef, 0)
    for i in 1 : length(model.net)
        try
            layer = model[i]
            push!(pa, trainable(layer)...)
        catch
            continue
        end
    end
    return Zygote.Params(pa)
end
