"""
    dataloader(name; header=true, dir=learninghorsedatasets)
Load a data for Machine Learning. `name` is either the name of the datasets or the full path of the data file to be loaded. The following three can be specified as the name of the dataset in `name`.
- `MNIST` : The MNIST Datasets
- `iris` : The iris Datasets
- `BostonHousing` : Boston Housing DataSets
And these datasets are downloaded and saved by creating a `dir` folder under the home directly(i.e. it is saved in the `/home_directory/learninghorsedatasets` by default).
When importing a data file, you can specify whether to read the header with `header`.

# Example
```@repl
julia> dataloader("MNIST");

julia> dataloader("/home/ubuntu/data/data.csv", header = false)
```
"""
function dataloader(name; header = true, dir = "learningdatasets")
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