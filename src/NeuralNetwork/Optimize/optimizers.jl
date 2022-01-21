export Descent, Momentum, AdaGrad, Adam

const ε = 1e-8

"""
    Descent(η=0.1)
Basic gradient descent optimizer with learning rate `η`.

# Parameters
- learning rate : `η`

# Example

"""
struct Descent
    eta::AbstractFloat
end

Descent(η::Float32 = 0.1) = Descent(η)

apply!(opt::Descent, x::AbstractArray, Δ::AbstractArray) = opt.eta .* Δ

"""
    Momentum(η=0.01, α=0.9, velocity)
Momentum gradient descent optimizer with learning rate `η` and parameter of velocity `α`.

# Parameters
- learning rate : `η`
- parameter of velocity : `α`

# Example

"""
struct Momentum
    eta::AbstractFloat
    alpha::AbstractFloat
    velocity::Dict
end

Momentum(η::Float32 = 0.01f0, α::Float32 = 0.9f0) = Momentum(η, α, Dict())

function apply!(opt::Momentum, x::AbstractArray, Δ::AbstractArray)
    η, α = opt.eta, opt.alpha
    v = get!(() -> zero(x), opt.velocity, x)::typeof(x) #get!() function returns a view.
    @. v = (α*v - η*Δ)
    @. Δ = -v
end

"""
    AdaGrad(η = 0.01)
Gradient descent optimizer with learning rate attenuation.

# Parameters
- η : initial learning rate

# Examples

"""
struct AdaGrad
    eta::AbstractFloat
    h::Dict
end

AdaGrad(η::Float32 = 0.01f0) = AdaGrad(η, Dict())

function apply!(opt::AdaGrad, x::AbstractArray, Δ::AbstractArray)
    η = opt.eta
    h = get!(() -> zero(x), opt.h, x)::typeof(x)
    @. h += Δ^2
    @. Δ *= η / (√h + ε)
    return Δ
end

"""
    Adam(η=0.01, β=(0.9, 0.99))
Gradient descent adaptive moment estimation optimizer.

# Parameters
- η : learning rate
- β : Decay of momentums

# Examples

"""
struct Adam
    eta::AbstractFloat
    beta::Tuple{AbstractFloat, AbstractFloat}
    recode::Dict
end

Adam(η::Float32 = 0.01f0, β::Tuple{Float32, Float32} = (0.9f0, 0.99f0)) = Adam(η, β, Dict())

function apply!(opt::Adam, x::AbstractArray, Δ::AbstractArray)
    η, β = opt.eta, opt.beta
    mt, vt , βp = get!(opt.recode, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end::Tuple{typeof(x),typeof(x),Vector{Float64}}
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ε) * η
    βp .= βp .* β    
    return Δ
end