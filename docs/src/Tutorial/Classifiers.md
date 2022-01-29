# Classifiers

## data processing
Prepare the data used in this chapter.
```
#load and processing the data
using HorseML.Preprocessing
data = Matrix(dataloader("iris"))
x, t = data[:, 1:4], data[:, 5]

LE = LabelEncoder()
t = LE(t)

scaler = Standard()
x = fit_transform!(scaler, x)

DS = DataSplitter(150, test_size = 0.3)

train_x, test_x = DS(x)
train_t, test_t = DS(t)

OHE = OneHotEncoder()
train_t = OHE(train_t)
```

## Logistic Regression
Let's create a classifier using the normalized Iris dataset. First of all, we will classify using Logistic Regression, a basic classifier.
```
using HorseML.Classification
using HorseML.Classification: fit!
model = Logistic()
fit!(model, train_x, train_t)

model(test_x)
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

accuracy_score(model, x, t) = mse(model(x), t)
accuracy_score(model, test_x, test_t)
```
Amazing! So far, we have been able to finish from building basic models of regression and classification to calculating accuracy.