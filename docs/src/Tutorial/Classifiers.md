# Classifiers

## Encoders
In order to convert label-like string data to teacher data that can be used by classifiers, you need to encode the data. The encoders can be used in the classification module.

### Label Encoder
First, convert string data to numeric data. At this time, the conversion rules for string data and numeric data are stored in the encoder. Therefore, after the prediction, you can see the prediction results by decoding with the same encoder. for more details, see [`LabelEncoder`](@ref).
```
using HorseML.Preprocessing
using HorseML.Classification
using HorseML.Classification: fit!, predict

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

## Logistic Regression
Let's create a classifier using the normalized Iris dataset. First of all, we will classify using Logistic Regression, a basic classifier.
```
model = Logistic()
fit!(model, train_x, train_t)

predict(model, test_x)
```

## SVC(Support Vector Machine Classifier)
When there are `N` classes, this classifier creates two-value classifiers that divides into one class and the other `N-1` classes.
at the time of prediction, all the classifiers generated predict, and the class that is predicted to have the highest probability of one class is the prediction result(This algorithm is called `One-vs-Rest`).
Mutiple classifiers are created, but the code is not much different from Logistic Regression.
```
model = SVC()
fit!(model, train_x, train_t)
```

# Accuracy Score
We've only trained and predicted until now, but it's necessary to know the accuracy when training models.
Here, let's use the LossFunction module to know the accuracy score. We haven't just predicted the SVC model yet, let's use it.
```
using HorseML.LossFunction

accuracy_score(model, x, t) = mse(predict(model, x), t)
accuracy_score(model, test_x, test_t)
```
Amazing! So far, we have been able to finish from building basic models of regression and classification to calculating accuracy.