function sample(range::UnitRange{Int64}, n; nc::Bool = false)
    N = range.stop - range.start + 2
    result = Array{Int64}(undef, n)
    m = 1
    ncm = 1
    t = 1
    if nc
        nc_result = Array{Int64}(undef, N-n-1)
        while m <= n
            if (N-t) * rand() >= (n - m+1)
                nc_result[ncm] = t
                ncm+=1
            else
                result[m] = t
                m+=1
            end
            t+=1
        end
        nc_result[ncm:end] = collect(t:range.stop)
        return result, nc_result
    else
        while m <= n
            if (N-t) * rand() < (n - m+1)
                result[m] = t
                m+=1
            end
            t+=1
        end
    
        return result
    end
end

macro dataframe_func(f)
    if isa(f, Expr) && (f.head == :function || is_short_function_def(f))
        definition = copy(f.args[1])
        body = copy(f.args[2])
        definition.head != :call && error("couldn't process arguments properly")
        for i in 2 : length(definition.args)
            if definition.args[i].head == :(::)
                try
                    if eval(definition.args[i].args[2]) <: AbstractArray
                        definition.args[i].args[2] = :DataFrame
                        sym = definition.args[i].args[1]
                        converter = :($sym = $sym |> Matrix)
                        insert!(body.args, findfirst(x -> typeof(x) == Expr, body.args), converter)
                    end
                catch e
                    !isa(e, UndefVarError) && rethrow(e)
                end
            end
        end
        return Expr(
            :escape,
            Expr(
                :block,
                Expr(:function, f.args[1], f.args[2]),
                Expr(
                    :function,
                    definition,
                    body
                )
            )
        )
    else
        error("@dataframe_func must be used with function definition")
    end
end