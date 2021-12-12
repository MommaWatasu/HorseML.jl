include("optimizers.jl")

function update!(opt, x, g)
    x .-= apply!(opt, x, g)
end

function update!(opt, gs::Zygote.Grads, ps::Params)
    for p in ps
        isnothing(gs[p]) && continue
        update!(opt, p, gs[p])
    end
end

make_d_tuple(d::Tuple) = d
make_d_tuple(d) = tuple(d)

function train!(N, loss, data, opt)
    ps = params(N)
    for d in data
        try
            g = gradient(ps) do
                loss(make_d_tuple(d)...)
            end
            update!(opt, g, ps)
        catch
            @warn "you can't train by one data because something wrong with the data!"
        end
    end
end

"""
    @epochs n ex
This macro cruns `ex` `n` times. Basically this is useful for learning NeuralNetwork.

# Example
julia> a = 1

julia> @epochs 1000 a+=1
progress:1000/1000
julia>a
1001
"""
macro epochs(n, ex)
    :(for i in 1 : $(esc(n))
        progress = "progress:"*string(i)*"/"*string($(esc(n)))*"\r"
        print(progress)
        $(esc(ex))
    end)
end