function tree_write(file, DT::DecisionTree, edges::Array{String, 1}, i; return_i = false)
    tree = DT.tree
    s = []
    push!(s, (nothing, tree))
    while length(s) != 0
        old_name, v = pop!(s)
        name = "node"*string(i)
        println(file, name*" [")
        st = ""
        if v["left"] != nothing
            st = "X["*string(v["feature_id"])*"]<"*string(v["threshold"])*"\n"
        end
        samples = sum(v["class_count"])
        st *= "samples="*string(samples)*"\n"
        st *= "value="*string(v["class_count"])
        println(file, "label = \""*st*"\"")
        println(file, "];")
        if old_name != nothing
            if old_name != "node1"
                push!(edges, old_name*"->"*name*";")
            elseif name == "node2"
                push!(edges, "node1->node2[label = \"True\"];")
            else
                push!(edges, "node1->"*name*"[label = \"False\"];")
            end
        end
        i += 1
        if v["left"] != nothing
            push!(s, (name, v["right"]))
            push!(s, (name, v["left"]))
        end
        if return_i
            return i
        end
    end
end

"""
    MV(path, forest; rounded=false, bg="#ffffff", fc="#000000", nc="#ffffff", label="Tree", fs="18")
Make DicisionTree and RandomForest Visual(make a dot file, see also [Graphviz](https://graphviz.org/)). The arguments are as follows:
- `path` : The full path of the dot file. The suffix must be `.dot`.
- `forest` : The model.
- `rounded` : If `rounded` is `true`, the nodes will be rounded.
- `bg` : Background color, type of this must be `String`.
- `fc` : Font color, type of this must be `String`.
- `nc` : Node color, type of this must be `String`.
- `label` : The label of the graph.
- `fs` : Font size, type of this must be `String`.

# Example
```repl
julia> MV("/home/ubuntu/test.dot", model, rounded = true)
```
"""
function MV(path, tree::DecisionTree; rounded = false, bg = "#ffffff", fc = "#000000", nc="#ffffff", label = "Tree", fs = "18") # Model Visualization
    if path[length(path) - 3 : length(path)] != ".dot"
        throw(ArgumentError("The file path you passed is not prefixed with'dot'."))
    end
    open(path, "w") do file
        println(file, "digraph Tree_model{")
        println(file, "graph[")
        println(file, "label = \""*label*"\",")
        println(file, "bgcolor = \""*bg*"\",")
        println(file, "fontcolor = \""*fc*"\",")
        println(file, "fontsize = \""*fs*"\",")
        println(file, "style = \"filled\",")
        println(file, "margin = 0.2\n];")
        println(file, "node[")
        println(file, "shape = box,")
        if rounded
            println(file, "style = \"rounded, filled\",")
        end
        println(file, "fillcolor = \""*nc*"\"\n];")
        i = 1
        edges = String[]
        tree_write(file, tree, edges, i)
        for edge in edges
            println(file, edge)
        end
        print(file, "}")
    end
end

function MV(paths::Vector{String}, forest::RandomForest{N}; rounded = false, bg = "#ffffff", fc = "#000000", label = "Tree", fs = "18", nc="#ffffff") where {N}# Model Visualization
    if length(paths) != N
        throw(ArgumentError("the number of trees in RandomForest and length of `paths` must match!"))
    end
    for i in 1 : N
        MV(paths[i], forest.forest[i], rounded=rounded, bg=bg, fc=fc, label=label, fs=fs, nc=nc)
    end
end