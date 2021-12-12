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
    eta::Float64
end

Descent(;η::Float64 = 0.1) = Descent(η)

apply!(opt::Descent, x::Array, Δ::Array) = opt.eta .* Δ

"""
    Momentum(η=0.01, α=0.9, velocity)
Momentum gradient descent optimizer with learning rate `η` and parameter of velocity `α`.

# Parameters
- learning rate : `η`
- parameter of velocity : `α`

# Example

"""
struct Momentum
    eta::Float64
    alpha::Float64
    velocity::Dict
end

Momentum(η = 0.01, α = 0.9) = Momentum(η, α, Dict())

function apply!(opt::Momentum, x::Array, Δ::Array)
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
    eta::Float64
    h::Dict
end

AdaGrad(η = 0.01) = AdaGrad(η, Dict())

function apply!(opt::AdaGrad, x::Array, Δ::Array)
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
    eta::Float64
    beta::Tuple{Float64, Float64}
    recode::Dict
end

Adam(η = 0.01, β = (0.9, 0.99)) = Adam(η, β, Dict())

function apply!(opt::Adam, x::Array, Δ::Array)
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