mutable struct ProgressIO
    bar_len::Int64
    default::IO
    i::IO
    o::IO
    function ProgressIO()
        old_stdout = stdout
        O = open(".progress_out.txt", "w")
        I = open(".progress_out.txt", "r")
        new(0, old_stdout, I, O)
    end
end

function change_to_default(PO::ProgressIO)
    redirect_stdout(PO.default)
end

function change_to_file(PO::ProgressIO)
    redirect_stdout(PO.o)
end

function end_of_progress(PO::ProgressIO)
    redirect_stdout(PO.default)
    close(PO.o)
    close(PO.i)
    try
        rm(".progress_out.txt")
    catch e
        throw(SystemError("couldn't remove `.progress_out.txt`!"))
    end
end

function show_logs(PO::ProgressIO)
    flush(PO.o)
    print(" "^PO.bar_len*"\r")
    print(read(PO.i, String))
end

"""
    @epochs n ex
This macro cruns `ex` `n` times. Basically this is useful for learning NeuralNetwork.
Even if there are any output during the progress, the progress bar won't disappear! It is always displayed in the bottom line of the output.
When the process is finished, display `Complete!`.
!!! note
    The output during training is displayed, but in order to keep displaying the progress bar, it is displayed collectively after each process.(This may be improved in the future)
!!! warning
    This macro may not work on Windows (because Windows locks files)! Use [`@simple_epochs`](@ref) instead!
# Example
```jldoctest
julia> function yes()
           println("yes")
           sleep(0.1)
       end
yes (generic function with 1 method)
julia> @epochs 10 yes()
yes
yes
yes
yes
yes
yes
yes
yes
yes
yes
  Complete!
```
"""
macro epochs(n, ex)
    quote
        PO = ProgressIO()
        for i in 1 : $(esc(n))
            progress = "progress:"*string(i)*"/"*string($(esc(n)))*"\r"
            print(progress)
            PO.bar_len = length(progress)
            change_to_file(PO)
            $(esc(ex))
            change_to_default(PO)
            show_logs(PO)
        end
        end_of_progress(PO)
        println("\033[92m \033[1m Complete! \033[m")
    end
end

"""
@simple_epochs n ex
It's not much different from `@epochs`, but it doesn't have the ability to keep the progress bar displayed.
# Example
```jldoctest
julia> function yes()
           println("yes")
           sleep(0.1)
       end
yes (generic function with 1 method)
julia> @epochs 10 yes()
yes
yes
yes
yes
yes
yes
yes
yes
yes
yes
  Complete!
```
"""
macro simple_epochs(n, ex)
    quote
        for i in 1 : $(esc(n))
            progress = "progress:"*string(i)*"/"*string($(esc(n)))*"\r"
            print(progress)
            flush(stdout)
            $(esc(ex))
        end
        println("\033[92m \033[1m Complete! \033[m")
    end
end