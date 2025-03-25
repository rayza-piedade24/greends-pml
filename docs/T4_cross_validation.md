The core part of the script is:

```
scores = cross_val_score(estimator=pipe_lr,
                           X=X_train,
                           y=y_train,
                           cv=10,
                           n_jobs=1)
```
