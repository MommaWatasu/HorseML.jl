function check_size(x, t)
    xs, ts = size(x, 2), length(t)
    if xs != ts
        throw(DimensionMismatch("first dimension of the input data is $xs, and teaching data is $ts. these dimensions must match."))
    end
end

expand(x::AbstractArray) = vcat(x, ones(1, size(x, 2)))