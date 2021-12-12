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