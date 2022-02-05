mutable struct GMM
    π::Vector{Float64}
    μ::Matrix{Float64}
    Σ::Array{Float64, 3}
    max::Int64
    th::Float64
    GMM(; max = 1e+8, th = 1e-4) = new(Array{Float64}(undef, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0, 0), max, th)
end

function initialize(x, K, N)
    d = x[:, 1]
    ma = maximum(d)
    mi = minimum(d)
    interval = (ma-mi) / K
    γ = zeros(N, K)
    for i in 1 : K-1
        γ[findall(x -> mi <= x < mi+interval, d), i] .= 1
        mi += interval
    end
    γ[findall(x -> mi < x <= mi+interval, d), K] .= 1
    println(γ)
    Mstep(x, γ)
end

function gauss(x, μ, σ)
    N, D = size(x)
    c1 = 1 / (2 * π)^(D/2)
    c2 = 1 / det(σ)^0.5
    c3 = x .- μ'
    c4 = c3 * σ'
    c5 = zeros(N)
    for d in 1 : D
        c5 += c4[:, d] .* c3[:, d]
    end
    p = @. c1 * c2 * exp(-c5 / 2)
    return p
end

function Estep(x, π, μ, Σ, N, K)
    y = zeros(N, K)
    for k in 1 : K
        y[:, k] = gauss(x, μ[k, :], Σ[k, :, :])
    end
    γ = zeros(N, K)
    for n in 1 : N
        wk = [π[k] * y[n, k] for k in 1 : K]
        γ[n, :] = wk / sum(wk)
    end
    return γ
end

function Mstep(x, γ, N, D, K)
    π = sum(γ, dims = 1)
    μ = zeros(K, D)
    for k in 1 : K
        μ[k, :] = γ[:, k]' * (x / sum(γ[:, k]))
    end
    Σ = zeros(K, D, D)
    for k in 1 : K
        for n in 1 : N
            wk = x .- μ[k, :]'
            wk = wk[n, :, :]
            Σ[k, :, :] += γ[n, k] * (wk * wk')
        end
        Σ[k, :, :] /= sum(γ[:, k])
    end
    return π, μ, Σ
end

function fit!(model::GMM, x, K)
    coverge(x, th) = @. abs(x) < th
    function coverge(π, μ, Σ, th)
        πb = coverge.(π, th) == trues(size(π)...)
        μb = coverge.(μ, th) == trues(size(μ)...)
        Σb = coverge.(Σ, th) == trues(size(Σ)...)
        πb && μb && Σb
    end
    N, D = size(x)
    π, μ, Σ = initialize(x, K, N)
    for _ in 1 : model.max
        γ = Estep(x, π, μ, Σ, N, K)
        π_new, μ_new, Σ_new = Mstep(x, γ, N, D, K)
        if coverge(π_new - π, μ_new - μ, Σ_new - Σ, model.th)
            #println(size(π_new), size(μ_new), size(Σ_new))
            model.π, model.μ, model.Σ = vec(π_new), μ_new, Σ_new
            return
        end
        π, μ, Σ = π_new, μ_new, Σ_new
    end
    @warn "Not Converged!"
    model.π, model.μ, model.Σ = vec(π), μ, Σ
end

function (model::GMM)(x)
    K = length(model.π)
    N = size(x, 1)
    Estep(x, model.π, model.μ, model.Σ, N, K)
end