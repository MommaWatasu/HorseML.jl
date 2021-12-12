"""
    Standard()
Standard Scaler. This scaler scale data as:
``\tilde{\boldsymbol{x}} = \\frac{x_{i}-\\mu}{\\sigma}``

# Example
```jldoctest preprocessing
julia> x = [
    16.862463771320925 68.10823385851712
    15.382965696961577 65.4313485700859
    8.916228406218375 53.92034559524475
    10.560285659132695 59.17305391117168
    12.142253214135884 62.28708207525656
    5.362107221163482 43.604947901567414
    13.893239446341777 62.44348617377496
    11.871357065173395 60.28433066289655
    29.83792267802442 69.22281924803998
    21.327107214235483 70.15810991597944
    23.852372696012498 69.81780163668844
    26.269031430914108 67.61037566099782
    22.78907104644012 67.78105545358633
    26.73342178134947 68.59263965946904
    9.107259141706415 56.565383817343495
    29.38551885863976 68.1005579469209
    7.935966787763017 53.76264777936664
    29.01677894379809 68.69484161138638
    6.839609488194577 49.69794758177567
    13.95215840314148 62.058116579899085]; #These data are also used to explanations of other functions.

julia> t = [169.80980778351542, 167.9081124078835, 152.30845618985222, 160.3110300206261, 161.96826472170756, 136.02842285615077, 163.98131131382686, 160.117817321485, 172.22758529098235, 172.21342437006865, 171.8939175591617, 169.83018083884602, 171.3878062674257, 170.52487535026015, 156.40282783981309, 170.6488327896672, 151.69267899906185, 172.32478221316322, 145.14365314788827, 163.79383292080666];

julia> scaler = Standard()
Standard(Float64[])

julia> fit!(scaler, x)
2×2 Matrix{Float64}:
 19.0591   64.467
  6.95818   6.68467

julia> transform!(scaler, x)
20×2 Matrix{Float64}:
  0.0310374   0.44579
  0.0104337   0.218714
  1.01139     0.710027
  1.37285     0.954091
 -0.895893   -0.270589
  0.983361    0.47397
  1.39855     0.645624
 -1.31861    -0.901482
  0.0147702   0.241708
  0.29675     0.483076
 -0.338824   -0.104812
  0.432442    0.387922
  0.860418    0.567095
 -0.495306    0.140871
  0.963084    0.552767
 -1.38901    -1.05926
  0.469323    0.719196
  0.0669475   0.512023
 -1.47583    -1.44017
 -1.99789    -3.27656
```
"""
mutable struct Standard
    p::AbstractVecOrMat
    Standard() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::Standard, x; dims=1)
    scaler.p = vcat(mean(x, dims=dims), std(x, dims=dims))
end

ss(x, m, s) = @. (x-m)/s

function transform!(scaler::Standard, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = ss(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = ss(x[i, :], p[:, i]...)
        end
    end
    return x
end

"""
    fit_transform!(scaler, x; dims=1)
fit scaler with `x`, and transform `x`.
"""
function fit_transform!(scaler::Standard, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x, dims=dims)
end

iss(x, m, s) = @. x*s+m

"""
    inv_transform!(scaler, x; dims=1)
Convert `x` in reverse.
"""
function inv_transform!(scaler::Standard, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = iss(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = iss(x[i, :], p[:, i]...)
        end
    end
    return x
end