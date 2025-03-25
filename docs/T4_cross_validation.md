
# Model Evaluation and Hyper-parameter Tuning (Mar 28, 2025)

## Combining transformers and estimators in a pipeline

In the previous notes where we discussed [sklearn pipelines](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md), the pipeline was created with `Pipeline`. There is, however an alternative that makes the code shorter is to use `make_pipeline` [see sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html). This is a shorthand for the `Pipeline constructor`; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically. The folowwing piece of code shows how to create a pipeline that scales the attributes and applies a logistic regression.

  ```
  pipe_lr = make_pipeline(StandardScaler(),
                          LogisticRegression())
  ```

The pipeline is then typically used in the following manner over train and test sets:

  ```
  pipe_lr.fit(X_train, y_train)
  y_pred = pipe_lr.predict(X_test)
  train_accuracy = pipe_lr.score(X_train, y_train) # accuracy estimate over the same data used for training
  test_accuracy = pipe_lr.score(X_test, y_test) # accuracy estimate over an independent test set
  ```

## Using k-fold cross-validation to assess model performance

The approach described above leads  in general to overfitting towards the train data set, and a bad performance over new examples. To prevent this, two diferent approaches can be followed

### The holdout method

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/holdout_method_fig62.png" alt="Alt Text" width="500" >

### Cross-validation

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/kfold_validation_fig_63.png" alt="Alt Text" width="500" >

In `sklearn`, cross-validation can easily be applied with [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html). Then, we can replace  `pipe_lr.fit(X_train, y_train)` in the script above by something like

  ```
  scores = cross_val_score(estimator=pipe_lr, # estimator with fit method
                             X=X_train,
                             y=y_train,
                             cv=10, # number of folds
                             n_jobs=1) # numbers of processors used (-1 for all processors)
  ```

that returns an array of scores of the estimator for each run of the cross validation. 

  ```
  from sklearn import datasets, linear_model
  from sklearn.model_selection import cross_val_score
  diabetes = datasets.load_diabetes()
  from sklearn.model_selection import StratifiedKFold
  X = diabetes.data[:150]
  y = diabetes.target[:150]
  lasso = linear_model.Lasso()
  skf = StratifiedKFold(n_splits=2)
  results = cross_val_score(lasso, X, y, cv=skf)
  ```


## Debugging algorithms with learning and validation curves

## Fine-tuning machine learning models via grid search

## Looking at different performance evaluation metrics



The core part of the script is:

```
scores = cross_val_score(estimator=pipe_lr,
                           X=X_train,
                           y=y_train,
                           cv=10,
                           n_jobs=1)
```
