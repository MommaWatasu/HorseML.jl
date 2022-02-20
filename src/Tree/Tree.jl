module Tree

include("../utils.jl")

export DecisionTree, MV, RandomForest

include("utils.jl")
include("DT.jl")
include("RF.jl")
include("MV.jl")

end