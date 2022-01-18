# LossFunction

`reduction` specifies the reduction to apply to the output:
- `none` : do nothing and return the vector of losses.
- `sum` : return the sum of the losses.
- `mean` : return the mean of the losses.

These functions take predictive values and teacher data as arguments, but only support the following types, respectively.
- Vector and Vector
- Vector and Number
- Matrix(just added an axis to the vector) and Vector
- Matrix(just added an axis to the vector) and Number
- Matrix and Matrix

```@docs
LossFunction.mse
LossFunction.cee
LossFunction.mae
LossFunction.huber
LossFunction.logcosh_loss
LossFunction.poisson
LossFunction.hinge
LossFunction.smooth_hinge
```