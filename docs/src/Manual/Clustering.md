# Clustering

```@meta
DocTestSetup = quote
    using CSV
    using DataFrames
    x = Array(CSV.read("../../../test/clustering.csv", DataFrame, header=false))
end
```

# Models
```@docs
Kmeans
GMM
```

```@meta
DocTestSetup = nothing
```