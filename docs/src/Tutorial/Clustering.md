# Clustering

## data processing
In clustering, if data is randomly generated, it will not converge, so use data that has already been confirmed to converge.
Get the test data with the following command:
```
$ wget https://github.com/MommaWatasu/HorseML.jl/raw/master/test/clustering.csv
```
Once you get it, let's load it with Julia.
```
using CSV
using DataFrames

df = CSV.read("clustering.csv", DataFrame, header=false)
x = Array(df)
```

## Kmeans method
First, let's perform clustering using the Kmeans method.
```
using HorseML.Clustering
using HorseML.Clustering: fit!

model = Kmeans(3)
fit!(model, x) # fitting
t = model(x) # predcting
```
Let's visualize the results.
```
using Plots

c1 = x[findall(t[:, 1].==1), :]
c2 = x[findall(t[:, 2].==1), :]
c3 = x[findall(t[:, 3].==1), :]

plot(c1[:, 1], c1[:, 2], seriestype = :scatter, title = "Clusters")
plot!(c2[:, 1], c2[:, 2], seriestype = :scatter)
plot!(c3[:, 1], c3[:, 2], seriestype = :scatter)
```
![Tree Visualized](../assets/clutering.png)
To find the error, you need to use a different function than regression.
```
using HorseML.LOssFunction

dm(x, model(x), model.μ)
```
The distortion measure should be used in this way as an exception.

## GMM(Gaussian Mixture Model)
Next, lets's  use Gaussian Mixture Model.
```
model = GMM(3)
fit!(model, x) # fitting
model(x) # predicting
```
The Gaussian Mixture Model also needs to use a different loss function.
```
π, μ, Σ = model.π, model.μ, model.σ
nlh(x, π, μ, Σ)
```