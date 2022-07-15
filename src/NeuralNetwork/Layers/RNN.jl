mutable struct RNN{W, WH, B, F}
    w::W
    wh::WH
    h
    b::B
    σ::F
end

function RNN(W::M, Wh::N, h, σ::F) where {M<:AbstractMatrix, N<:AbstractMatrix, F}
    b = create_bias(W, size(W, 1))
    RNN{M, N, typeof(b), F}(W, Wh, h, b, σ)
end

function RNN(io::Pair{<:Integer, <:Integer}, σ;
        initW=nothing, initb=nothing, set_w = "Xavier", high_accuracy::Bool=false)
    if initW != nothing
        if initb != nothing
            return RNN{typeof(initW), typeof(initb), typeof(σ)}(initW, initb, σ)
        else
            return RNN(initW, σ)
        end
    end
    w = dense_w(io..., set_w)
    out = io[2]
    wh = dense_w(out, out, set_w)
    RNN(w, wh, zeros(out), σ)
end

trainable(R::RNN) = D.w, D.b

function (R::RNN)(X::AbstractVecOrMat)
    W, b, h, Wh, σ = R.w, R.b, R.h, R.wh, R.σ
    R.h = σ.(muladd(Wh, h, muladd(W, X, b)))
end