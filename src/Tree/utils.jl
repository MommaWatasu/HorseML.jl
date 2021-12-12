function check_size(x, y)
    x_s, y_s = size(x, 1), size(y, 1)
    if x_s != y_s
        throw(DimensionMismatch("first dimension of the input data is $x_s, and teaching data is $y_s. these dimensions must match."))
    end
end