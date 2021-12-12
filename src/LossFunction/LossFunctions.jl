module LossFunction

using LinearAlgebra
using Statistics

LOSSES = [
    :mse, :cee, :mae, :huber, :logcosh_loss, :poisson, :hinge, :smooth_hinge
]

for f in LOSSES
    @eval export $(f)
end

safe_log(x, miniv = 1e-8) = map(log, clamp.(x, miniv, Inf))

@doc raw"""
    mse(y, t; reduction="mean")
Mean Square Error. This is the expression:
```math
MSE(y, t) = \frac{\sum_{i=1}^{n} (t_{i}-y_{i})^{2}}{n}
```
"""
function mse(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = (y.-t).^2
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end
mse(y::Number, t::Number) = (y-t)^2

@doc raw"""
    cee(y, t; reduction="mean")
Cross Entropy Error. This is the expression:
```math
CEE(y, t) = \frac{\sum_{i=1}^{n} t\ln y}{n}
```
"""
function cee(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = -t.*safe_log(y)
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

cee(y::Number, t::Number) = -t*safe_log(y)

@doc raw"""
    mae(y, t)
Mean Absolute Error. This is the expression:
```math
MAE(y, t) = \frac{\sum_{i=1}^{n} |t_{i}-y_{i}|}{n}
```
"""
function mae(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = abs.(t-y)
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

mae(y::Number, t::Number) = abs(t-y)

@doc raw"""
    huber(y, t; δ=1, reduction="mean")
Huber-Loss. If `δ` is large, it will be a function like [`mse`](@ref), and if it is small, it will be a function like [`mae`](@ref). This is the expression:
```math
a = |t_{i}-y_{i}| \\
Huber(y, t) = \frac{1}{n} \sum_{i=1}^{n} \left\{
\begin{array}{ll}
\frac{1}{2}a^{2} & (a \leq \delta) \\
\delta(a-\frac{1}{2}\delta) & (a \gt \delta)
\end{array}
\right.
```
"""
function huber(y::AbstractVector, t::AbstractVector; reduction::String="mean", δ=1)
    a = abs.(t-y)
    loss = @. ifelse(a<=δ, a^2/2, (a-δ/2)δ)
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

huber(y::Number, t::Number; δ=1) = ifelse(abs(t-y)<=δ, abs(t-y)^2/2, (abs(t-y)-δ/2)δ)

@doc raw"""
    logcosh_loss(y, t; reduction="mean")
Log Cosh. Basically, it's [`mae`](@ref), but if the loss is small, it will be close to [`mse`](@ref). This is the expression:
```math
Logcosh(y, t) = \frac{\sum_{i=1}^{n} \log(\cosh(t_{i}-y_{i}))}{n}
```
"""
function logcosh_loss(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = @. log(cosh(t-y))
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

logcosh_loss(y::Number, t::Number) = log(cosh(t-y))

@doc raw"""
    Poisson(y, t; reduction="mean")
Poisson Loss, Distribution of predicted value and loss of Poisson distribution. This is the expression:
```math
Poisson(y, t) = \frac{\sum_{i=1}^{n} y_{i}-t_{i} \ln y}{n}
```
"""
function poisson(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = @. y - t*log(y)
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

poisson(y::Number, t::Number) = y - t*log(y)

@doc raw"""
    hinge(y, t; reduction="mean")
Hinge Loss, for SVM. This is the expression:
```math
Hinge(y, t) = \frac{\sum_{i=1}^{n} \max(1-y_{i}t_{i}, 0)}{n}
```
"""
function hinge(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    loss = @. max(1-y*t, 0)
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

hinge(y::Number, t::Number) = max(1-y*t, 0)

@doc raw"""
    smooth_hinge(y, t; reduction="mean")
Smoothing Hinge Loss. This is the expression:
```math
smoothHinge(y, t) = \frac{1}{n} \sum_{i=1}^{n} \left\{
\begin{array}{ll}
0 & (t_{i}y_{i} \geq 1) \\
\frac{1}{2}(1-t_{i}y_{i})^{2} & (0 \lt t_{i}y_{i} \lt 1) \\
\frac{1}{2} - t_{i}y_{i} & (t_{i}y_{i} \leq 0)
\end{array}
\right.
```
"""
function smooth_hinge(y::AbstractVector, t::AbstractVector; reduction::String="mean")
    z = t.*y
    loss = similar(z)
    for i in 1 : length(y)
        if z[i] >= 0
            loss[i] = 0
        elseif 0 < z[i] < 1
            loss[i] = (1-z[i])^2/2
        else
            loss[i] = 1/2-z[i]
        end
    end
    if reduction=="none"
        return loss
    elseif reduction=="sum"
        return sum(loss)
    elseif reduction=="mean"
        return mean(loss)
    else
        throw(ArgumentError("`reduction` must be either `none`, `sum` or `mean`"))
    end
end

function smooth_hinge(y::Number, t::Number)
    if t*y >= 0
        return 0
    elseif 0 < t*y < 1
        return (1-t*y)^2/2
    else
        return 1/2-t*y
    end
end

#TODO:Consider whether to implement Calback liverer information volume

for lossfunc in LOSSES
    #For Vector and Number
    @eval begin
        function $(lossfunc)(y::AbstractVector{T}, t::Number; reduction::String="mean") where {T}
            $(lossfunc)(y, fill(t, length(y)), reduction=reduction)
        end
    end
    #For Matrix and Number
    @eval begin
        function $(lossfunc)(y::AbstractMatrix{T}, t::Number; reduction::String="mean") where {T}
            if length(y) == 1
                return $(lossfunc)(y..., t)
            elseif size(y, 1)==1 || size(y, 2)==1
                return $(lossfunc)(vec(y), t, reduction=reduction)
            else
                throw(DimensionMismatch("LossFunctions don't support for Matrix!"))
            end
        end
    end
end

end