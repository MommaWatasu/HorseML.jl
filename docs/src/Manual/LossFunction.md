# LossFunction

`reduction` specifies the reduction to apply to the output:
- `none` : do nothing and return the vector of losses.
- `sum` : return the sum of the losses.
- `mean` : return the mean of the losses.

These functions must have both the input `y` and `t` as vectors.

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