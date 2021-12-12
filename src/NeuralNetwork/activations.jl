ACTIVATIONS = [
    :line, :σ, :hardσ, :hardtanh, :relu,
    :leakyrelu, :relu6, :rrelu, :elu, :gelu, :swish, :selu,
    :celu, :softplus, :softsign, :logσ, :logcosh,
    :mish, :tanhshrink, :softshrink, :trelu, :lisht 
]
for f in ACTIVATIONS
    @eval export $(f)
end

"""
    line(x)
Linear function. This is the expression:
```math
line(x) = x
```
"""
@inline line(x) = x

@doc raw"""
    σ(x)
Standard sigmoid activation function. Also, this function can be called with `σ`. This is the expression:
```math
\sigma(x) = \frac{1}{1+e^{-x}}
```
"""
@inline function sigmoid(x)
    t = exp(-abs(x))
    return (x>=0) ? inv(1 + t) : t / (t + 1)
end
const σ = sigmoid

@doc raw"""
    hardsigmoid(x) = max(0, min(1, (x + 2.5) / 6))
Piecewise linear approximation of sigmoid. Also, this function can be called with `hardσ`. This is the expression:
```math
hardsigmoid(x) = \left\{
\begin{array}{ll}
1 & (x \geq \frac{1}{4}) \\
\frac{1}{5} x & (- \frac{1}{4} \lt x \lt \frac{1}{4}) \\
0 & (x \leq - \frac{1}{4})
\end{array}
\right.
```
"""
@inline hardsigmoid(x) = max(0, min(1, (x + 2.5) / 6))
const hardσ = hardsigmoid

@doc raw"""
    lisht(x)
This is the expression:
```math
lisht(x) = x\tanh(x)
```
"""
@inline lisht(x) = x*tanh(x)

@doc raw"""
    relu(x) = max(0, x)
`relu` is `Rectified Linear Unit`. This is the expression:
```math
relu(x) = \left\{
\begin{array}{ll}
x & (x \geq 0) \\
0 & (x \lt 0)
\end{array}
\right.
```
"""
@inline relu(x) = max(0, x)

@doc raw"""
    leakyrelu(x; α=0.01) = (x>0) ? x : α*x
Leaky Rectified Linear Unit. This is the expression:
```math
leakyrelu(x) = \left\{
\begin{array}{ll}
\alpha x & (x \lt 0) \\
x & (x \geq 0)
\end{array}
\right.
```
"""
@inline leakyrelu(x; α=0.01) = max(x, α*x)

@doc raw"""
    rrelu(min, max)
Randomized Rectified Linear Unit. The expression is the as [`leakyrelu`](@ref). but `α` is a random number between `min` and `max`.
Also, since this function is defined as a structure, use it as follows:
```
Dense(10=>5, rrelu(0.001, 0.1))
```
"""
struct rrelu
    α::Float64
    function rrelu(min::Float64, max::Float64)
        new(round(rand()*(max-min+1))+min)
    end
end

@inline (rrelu::rrelu)(x) = max(x, rrelu.α*x)

@doc raw"""
    relu6(x)
Relu function with an upper limit of 6. This is the expression:
```math
relu6(x) = \left\{
\begin{array}{ll}
6 & (x \gt 6) \\
x & (x \geq 0) \\
0 & (x \lt 0)
\end{array}
\right.
```
"""
@inline relu6(x) = min(6, max(0, x))

@doc raw"""
    elu(x, α=1)
Exponential Linear Unit activation function. You can also specify the coefficient explicitly, e.g. elu(x, 1). This is the expression:
```math
elu(x, α) = \left\{
\begin{array}{ll}
x & (x \geq 0) \\
\alpha(e^x-1) & (x \lt 0)
\end{array}
\right.
```
"""
@inline elu(x; α=1) = (x<0) ? (exp(x)-1)α : x

@doc raw"""
    celu(x; α=1)
Continuously Differentiable Exponential Linear Unit. This is the expression:
```math
\alpha = 1 \\
celu(x) = \left\{
\begin{array}{ll}
x & (x \geq 0) \\
\alpha(e^\frac{x}{\alpha}-1) & (x \lt 0)
\end{array}
\right.
```
"""
@inline celu(x; α=1) = (x>=0) ? x : α*(exp(x/α)-1)

@doc raw"""
    gelu(x)
Gaussian Error Linear Unit. This is the expression(``\phi`` is a distribution function of standard normal distribution.):
```math
gelu(x) = x\phi(x)
```
However, in the implementation, it is calculated with the following expression.
```math
\sigma(x) = \frac{1}{1+e^{-x}} \\
gelu(x) = x\sigma(1.702x)
```
"""
@inline gelu(x) = σ(1.702*x)x

@doc raw"""
    selu(x)
Scaled exponential linear units. This is the expression
```math
\lambda = 1.0507009873554804934193349852946 \\
\alpha = 1.6732632423543772848170429916717 \\
selu(x) = \lambda \left\{
\begin{array}{ll}
x & (x \geq 0) \\
\alpha(e^x-1) & (x \lt 0)
\end{array}
\right.
```
"""
@inline function selu(x)
    λ = oftype(float(x), selu_λ)
    α = oftype(float(x), selu_α)
    return ((x > 0) ? x : (exp(x)-1)α)*λ
end

const selu_λ = 1.0507009873554804934193349852946
const selu_α = 1.6732632423543772848170429916717

@doc raw"""
    trelu(x; θ=1)
Threshold gated Rectified Linear Unit. This is the expression:
```math
\theta = 1 \\
trelu(x) = \left\{
\begin{array}{ll}
x & (x \gt 0) \\
0 & (x \leq 0)
\end{array}
\right.
```
"""
@inline trelu(x; θ=1) = (x<=θ) ? 0 : x

@doc raw"""
    logσ(x)
logarithmic sigmoid function. This is the expression:
```math
\sigma(x) = \frac{1}{1+e^{-x}} \\
logsigmoid(x) = \log(\sigma(x))
```
"""
@inline logsigmoid(x) = log(σ(x))
const logσ = logsigmoid

@doc raw"""
    logcosh(x)
Log-Cosh function. This is the expression:
```math
logcosh(x) = \log(\cosh(x))
```
"""
@inline logcosh(x) = log(cosh(x))

@doc raw"""
    hardtanh(x)
Linear tanh function. This is the expression:
```math
hardtanh(x) = \left\{
\begin{array}{ll}
1 & (x \geq 1) \\
x & (-1 \lt x \lt 1) \\
-1 & (x \leq -1)
\end{array}
\right.
```
"""
@inline hardtanh(x) = min(1, max(-1, x))

@doc raw"""
    tanhshrink(x)
Shrink tanh function. This is the expression:
```math
tanhshrink(x) = 1-\tanh(x)
```
"""
@inline tanhshrink(x) = 1-tanh(x)

@doc raw"""
    softshrink(x; λ=0.5)
This is the expression:
```math
\lambda=0.5 \\
softshrink(x) = \left\{
\begin{array}{ll}
x-\lambda & (x \gt \lambda) \\
0 & (-\lambda \leq x \leq \lambda) \\
x+\lambda & (x \lt -\lambda) \\
\end{array}
\right.
```
"""
@inline softshrink(x; λ=0.5) = min(x+λ, max(x-λ, 0))

@doc raw"""
    softsign(x) = x / (1+abs(x))
The softsign activation function. This is the expression:
```math
softsign(x) = \frac{x}{1+|x|}
```
"""
@inline softsign(x) = x / (1 + abs(x))

@doc raw"""
    softplus(x) = log(1 + exp(x))
the softplus activation function. This is the expression:
```math
softplus(x) = \ln(1+e^x)
```
"""
@inline softplus(x) = log(1 + exp(x))

@doc raw"""
    mish(x) = x * tanh(softplus(x))
The mish function. This is the expression:
```math
softplus(x) = \ln(1+e^x) \\
mish(x) = x\tanh(softplus(x))
```
"""
@inline mish(x) = x * tanh(softplus(x))

@doc raw"""
    swish(x; β=1)
The swish function. This is the expression:
```math
\sigma(x) = \frac{1}{1+e^{-x}} \\
swish(x) = x\sigma(\beta x)
```
"""
@inline swish(x; β=1) = x * σ(β*x)
