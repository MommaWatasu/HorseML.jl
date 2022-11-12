"""
    dataloader(name; header=true, dir="HorseMLdatasets")
Load a data for Machine Learning. `name` is either the name of the datasets or the full path of the data file to be loaded. The following three can be specified as the name of the dataset in `name`.
- `MNIST` : The MNIST Datasets
- `iris` : The iris Datasets
- `BostonHousing` : Boston Housing DataSets
And these datasets are downloaded and saved by creating a `dir` folder under the home directly(i.e. it is saved in the `/home_directory/HorseMLdatasets` by default).
When importing a data file, you can specify whether to read the header with `header`.

# Example
```@repl
julia> dataloader("MNIST");

julia> dataloader("/home/ubuntu/data/data.csv", header = false)
```
"""
function dataloader(name; header = true, dir = "HorseMLdatasets")
    hd = homedir()
    dir = joinpath(hd, dir)
    mkpath(dir)
    current = pwd()
    cd(dir)
    if name == "MNIST"
        if !(isfile("mnist_train.csv"))
            yn = Base.prompt("Can I download the mnist_train.csv(110MB)? (y/n)")
            yn == "y" && Downloads.download("https://pjreddie.com/media/files/mnist_train.csv", "mnist_train.csv")
        end
        df_r = CSV.read("mnist_train.csv", header = false, DataFrame)
        if !(isfile("mnist_test.csv"))
            yn = Base.prompt("Can I download the mnist_test.csv(18MB)? (y/n)")
            yn == "y" && Downloads.download("https://pjreddie.com/media/files/mnist_test.csv", "mnist_test.csv")
        end
        df_e = CSV.read("mnist_test.csv",header = false , DataFrame)
        cd(current)
        return df_r, df_e
    elseif name == "iris"
        url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
        if !(isfile("iris.csv"))
            yn = Base.prompt("Can I download the iris.csv(4KB)? (y/n)")
            yn == "y" && Downloads.download(url, "iris.csv")
        end
        df = CSV.read("iris.csv", DataFrame)
        cd(current)
        return df
    elseif name == "BostonHousing"
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        if !(isfile("BostonHousing.csv"))
            yn = Base.prompt("Can I download the BostonHousing.csv(36KB)? (y/n)")
            yn == "y" && Downloads.download(url, "BostonHousing.csv")
        end
        df = CSV.read("BostonHousing.csv", DataFrame)
        cd(current)
        return df
    end
    try
        df = CSV.read(name, header=header, DataFrame)
        return df
    finally
        cd(current)
    end
end

"""
    databuilder(x, y; batches=1)
`x` is the feature value, `y` is the teacher data, and `bacthes` is the batch size.
This function formats and returns the data used for the neural network(however, `x` and` y` must be Arrays or DataFrames, so please use this function  after encoding, normalization, etc.).

# Example
```jldoctest
julia> data = dataloader("iris");

julia> LE, OHE = LabelEncoder(), OneHotEncoder()
(LabelEncoder(Dict{Any, Any}()), OneHotEncoder())

julia> x, y = data[:, Not(:variety)], OHE(LE(data[:, :variety]));

julia> databuilder(x, y)
150-element Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}:
 ([5.1; 3.5; 1.4; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.9; 3.0; 1.4; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.7; 3.2; 1.3; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.6; 3.1; 1.5; 0.2;;], [1.0; 0.0; 0.0;;])
 ([5.0; 3.6; 1.4; 0.2;;], [1.0; 0.0; 0.0;;])
 ([5.4; 3.9; 1.7; 0.4;;], [1.0; 0.0; 0.0;;])
 ([4.6; 3.4; 1.4; 0.3;;], [1.0; 0.0; 0.0;;])
 ([5.0; 3.4; 1.5; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.4; 2.9; 1.4; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.9; 3.1; 1.5; 0.1;;], [1.0; 0.0; 0.0;;])
 ([5.4; 3.7; 1.5; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.8; 3.4; 1.6; 0.2;;], [1.0; 0.0; 0.0;;])
 ([4.8; 3.0; 1.4; 0.1;;], [1.0; 0.0; 0.0;;])
 ⋮
 ([6.0; 3.0; 4.8; 1.8;;], [0.0; 0.0; 1.0;;])
 ([6.9; 3.1; 5.4; 2.1;;], [0.0; 0.0; 1.0;;])
 ([6.7; 3.1; 5.6; 2.4;;], [0.0; 0.0; 1.0;;])
 ([6.9; 3.1; 5.1; 2.3;;], [0.0; 0.0; 1.0;;])
 ([5.8; 2.7; 5.1; 1.9;;], [0.0; 0.0; 1.0;;])
 ([6.8; 3.2; 5.9; 2.3;;], [0.0; 0.0; 1.0;;])
 ([6.7; 3.3; 5.7; 2.5;;], [0.0; 0.0; 1.0;;])
 ([6.7; 3.0; 5.2; 2.3;;], [0.0; 0.0; 1.0;;])
 ([6.3; 2.5; 5.0; 1.9;;], [0.0; 0.0; 1.0;;])
 ([6.5; 3.0; 5.2; 2.0;;], [0.0; 0.0; 1.0;;])
 ([6.2; 3.4; 5.4; 2.3;;], [0.0; 0.0; 1.0;;])
 ([5.9; 3.0; 5.1; 1.8;;], [0.0; 0.0; 1.0;;])
```
"""
function databuilder(x::Matrix{TX}, t::Matrix{TY}; batches = 1) where {TX, TY}
    PT = promote_type(TX, TY)
    x, t = Matrix{PT}(x), Matrix{PT}(t)
    size(x, 1) % batches != 0 && @warn "the data size is not divisible by the batch size! some data will be omitted"
    N = round(Int64, size(x, 1) / batches)
    x, t = transpose(x), transpose(t)
    data = Vector{Tuple{Matrix{TX}, Matrix{TY}}}(undef, N)
    c = 1
    for n in 1 : N
        data[n] = (x[:, c:n*batches], t[:, c:n*batches])
        c += batches
    end
    return data
end

function databuilder(x::Array{TX, 4}, t::Matrix{TY}; batches = 1) where {TX, TY}
    PT = promote_type(TX, TY)
    x, t = Array{PT, 4}(x), Matrix{PT}(t)
    size(x, 1) % batches != 0 && @warn "the data size is not divisible by the batch size! some data will be omitted"
    N = round(Int64, size(x, 1) / batches)
    x, t = permutedims(x, (2, 3, 4, 1)), transpose(t)
    data = Vector{Tuple{Array{PT, 4}, Matrix{PT}}}(undef, N)
    c = 1
    for n in 1 : N
        data[n] = (x[:, :, :, c:n*batches], t[:, c:n*batches])
        c += batches
    end
    return data
end

databuilder(x::Array, t::Vector; batches = 1) = databuilder(x, t[:, :], batches = batches)
databuilder(x::DataFrame, t; batches=1) = databuilder(Matrix(x), t, batches = batches)
databuilder(x::DataFrame, t::DataFrame; batches=1) = databuilder(Matrix(x), Matrix(t), batches = batches)

"""
    DataSplitter(ndata; test_size=nothing, train_size=nothing)
Split the data into test data and training data. `ndata` is the number of the data, and you must specify either `test_size` or `train_size`. thease parameter can be proprtional or number of data.

!!! note
    If both `test_size` and `train_size` are specified, `test_size` takes precedence.

#Example
```jldoctest
julia> x = rand(20, 2);

julia> DS = DataSplitter(50, train_size = 0.3);

julia> train, test = DS(x, dims = 2);

julia> train |> size
(6, 2)

julia> test |> size
(14, 2)
```
"""
struct DataSplitter
    #TODO: make it possible to DataFrame split
    test_indices::Array{Int64, 1}
    train_indices::Array{Int64, 1}
    function DataSplitter(ndata; test_size=nothing, train_size=nothing)
        test_size == train_size == nothing && throw(ArgumentError("test_size and train_size wasn't specified. you must specify either one."))
        if test_size == nothing
            #number of train data is ceiled
            test_size = (0<=train_size<=1) ? ndata-ceil(train_size*ndata) : ndata-train_size
        else
            test_size = (0<=test_size<=1) ? ndata-ceil((1-test_size)*ndata) : test_size
        end
        new(sample(1:ndata, Int(test_size), nc=true)...)
    end
end

function (DS::DataSplitter)(xs; dims=1)
    test_index = fill!(Array{Union{Colon, Array{Int64, 1}}}(undef, ndims(xs)), :)
    train_index = fill!(Array{Union{Colon, Array{Int64, 1}}}(undef, ndims(xs)), :)
    train_index[dims] = DS.train_indices
    test_index[dims] = DS.test_indices
    return xs[train_index...], xs[test_index...]
end

function make_circles(; n_samples::Union{Int, Tuple{Int, Int}}=100, shuffle::Bool=true, noise::Union{Nothing, Float64}=nothing, factor::Float64=0.5)
    if typeof(n_samples) == Int
        n_samples = (div(n_samples, 2), div(n_samples+1, 2))
    end
    θ1 = 0:2π/n_samples[1]:2π
    θ2 = 0:2π/n_samples[2]:2π
    data = [hcat(sin.(θ1), cos.(θ1)); hcat(sin.(θ2), cos.(θ2)).*factor]
    labels = [zeros(Int, n_samples[1]); ones(Int, n_samples[2])]
    if noise != nothing
        data+=rand(size(data)...).*noise
    end
    if shuffle
        idx = collect(1:sum(n_samples))
        shuffle!(idx)
        return (data[idx, :], labels[idx])
    end
    return data
end