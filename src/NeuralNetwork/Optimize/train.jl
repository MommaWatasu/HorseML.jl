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
