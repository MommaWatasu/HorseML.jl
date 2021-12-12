"""
    Ridge(alpha = 0.1)
Ridge Regression. `alpha` is the value multiplied by regularization term.

# Example
```jldoctest regression
julia> model = Ridge()
Ridge(Float64[], 0.1)

julia> fit!(model, x, t)
3-element Vector{Float64}:
 -0.5635468573581848
  2.1185952951687614
 40.334109796666425

julia> model(x)
20-element Vector{Float64}:
 175.12510514593038
 170.28763505842625
 149.54478779081344
 159.7466476176333
 165.4525001910219
 129.6935485937018
 164.79709438985097
 161.3621431448216
 170.17418141858434
 176.95192713546982
 174.8078461898064
 168.76930346791391
 171.09202561187362
 170.58861763111338
 155.04089855305028
 168.05151465675456
 149.76311329450505
 169.5183634524783
 141.7695072903308
 163.94744858842117
```
"""
mutable struct Ridge
    w::Array
    α::Float64
    Ridge(alpha = 0.1) = new(Array{Float64}(undef, 0), alpha)
end

function fit!(model::Ridge, x, t)
    check_size(x, t)
    x = expand(x)
    n = size(x, 1)
    _I = Matrix{Float64}(I, n, n)
    model.w = inv(x * x' .+ model.α * _I) * x * t
end

(model::Ridge)(x) = expand(x)' * model.w