# Preprocessing

## Data Loader
With Preprocessing module, you can load the data. Also, some famous data sets can be loaded by specifying a name.
First, let's load the MNIST data.
```
using HorseML.Preprocessing

train_data, test_dara = dataloader("MNIST")
```
If it is your first time to load a data set. you wil be asked if you want to download the data like this:
```
Can I download the mnist_train.csv(110MB)? (y/n)
```

!!! note
    The data downloaded here is in `/home_directory/HorseMLdatasets/` by default. for more details, see [`dataloader`](@ref).

Let's also read the local data. 
```
#Please create train.csv directly under the /home_directory/learningdatasets/ as follows:
#0.202027,0.246752,0.608301,0.406351,0.0260402,0.747845
#0.735292,0.332892,0.438458,0.0787028,0.797796,0.294831
#0.710725,0.213594,0.527118,0.579191,0.298599,0.23684
#0.288168,0.787194,0.809412,0.464031,0.960465,0.655897
df = dataloader("/home_directory/HorseMLdatasets/train.csv", header = false)
```

## Data Preprocessing

### Normalization
Even if you use the data as it is, you won't be able to make a model with high accuracy, let's normalize the data.
The following three scalers are available in HorseML:
- Standard Scaler : ``\tilde{\boldsymbol{x}} = \frac{x_{i}-\mu}{\sigma}``
- MinMax Scaler : ``\tilde{\boldsymbol{x}} = \frac{\boldsymbol{x}-min(\boldsymbol{x})}{max(\boldsymbol{x})-min(\boldsymbol{x})}``
- Robust Scaler : ``\tilde{\boldsymbol{x}} = \frac{\boldsymbol{x}-Q2}{Q3 - Q1}``
For example, if you normalize iris data set with Standard Scaler,
```
using HorseML.Preprocessing: fit!
data = Matrix(dataloader("iris"))
x, t = data[:, 1:4], data[:, 5]

scaler = Standard()
x = fit_transform!(scaler, x)
```
Once the scaler is fittted, sclaes using the same value unless it fits again.

### Data Spliting
Data division is done by [`DataSplitter`](@ref).
!!! warning
    this function is broken in HorseML v0.3.2
```
DS = DataSplitter(150, test_size = 0.3)

train_x, test_x = DS(x)
train_t, test_t = DS(t)
```
You can specify `test_size` and `train_size`, whether it's the number of data or the percentage.

## Encoding
In order to convert label-like string data to teacher data that can be used by classifiers, you need to encode the data. The encoders can be used in the classification module.

### Label Encoder
First, convert string data to numeric data. At this time, the conversion rules for string data and numeric data are stored in the encoder. Therefore, after the prediction, you can see the prediction results by decoding with the same encoder. for more details, see [`LabelEncoder`](@ref).
```
LE = LabelEncoder()
test_t = LE(test_t)#This data is splitted in the previous chapter.
train_t = LE(train_t)
```

### One-Hot Encoder
Next, let's ocnvert the data into One-Hot formst for make learning easier. for more details, see [`OneHotEncoder`](@ref).
```
OHE = OneHotEncoder()
test_t = OHE(test_t)
train_t = OHE(train_t)
```