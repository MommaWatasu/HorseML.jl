function check_size(x, t)
    xs, ts = size(x, 1), length(t)
    if xs != ts
        throw(DimensionMismatch("first dimension of the input data is $xs, and teaching data is $ts. these dimensions must match."))
    end
end

expand(x::AbstractArray) = hcat(x, ones(size(x, 1), 1))