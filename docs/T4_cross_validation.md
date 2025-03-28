
# Model Evaluation and hyper-parameter Tuning

<details markdown="block">
<summary> Combining transformers and estimators in a pipeline </summary>

## Combining transformers and estimators in a pipeline

In the previous notes where we discussed [sklearn pipelines](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md), the pipeline was created with `Pipeline`. There is, however an alternative that makes the code shorter is to use `make_pipeline` [see sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html). This is a shorthand for the `Pipeline constructor`; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically. The following piece of code shows how to create a pipeline that scales the attributes and applies a logistic regression.

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
---

</details>

<details markdown="block">
<summary> Using k-fold cross-validation to assess model performance </summary>

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

that returns an array of scores of the estimator for each run of the cross validation. The parameter `cv` can be used to indicate which cross validation scheme should be used. It could take for instance one of the following: 

- [KFold](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold): divides all the samples in groups of samples, called folds. This is equivalent to just use, e.g., `cv=10`.
- [GroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-group-k-fold): this a variation of k-fold which ensures that the same group is not represented in both testing and training sets
- [StratifiedKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold): this is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-group-k-fold): The idea is to try to preserve the distribution of classes in each split while keeping each group within a single split.

For instance, the following code stratifies folds by the target class `y`. So, if for instance there are 100 examples of class 0 and 10 examples of class 1, then all folds get 20 examples from class 0 and 2 examples for class 1 (since `n_splits=5`).

  ```
  # model
  clf = DecisionTreeClassifier(max_depth=10)
  # cv strategy 
  skf = StratifiedKFold(n_splits=5)
  # fit and predict over the validation set
  results = cross_val_score(clf, X_train, y_train, cv=skf)
  ```

**Script** to read italian wine regions data from the UCI repository, and applies stratified croass validation to predict the region from the wine attributes: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/cross_val_score_stratifiedkfold.ipynb)

---

</details>

<details markdown="block">
<summary> Learning and validation curves </summary>


## Learning and validation curves

A [learning curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve) shows cross-validated training and test scores for different training set sizes.

A [validation curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve) determine training and test scores for varying parameter values. This is equivalent to grid search (see below) for a single parameter.

**Script** to read italian wine region data and create learning curve for a given classifier: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/learning_curve.ipynb)

---

</details>

<details markdown="block">
<summary>Tuning machine learning hyper-parameters via grid search </summary>

## Tuning machine learning hyper-parameters via grid search

One of the most critical steps in machine learning is tuning hyper-parameters of the model, e.g. `max_depth` for a decision tree. It is possible and recommended to search the hyper-parameter space for the best cross validation score. See [sklearn grid search section](https://scikit-learn.org/stable/modules/grid_search.html#grid-search).

A search consists of:
- an estimator: regressor or classifier such as `sklearn.tree.DecisionTreeClassifier()`;
- a parameter space such as `param_grid = [{'max_depth': [4,5,6,7]}]`;
- a method for searching or sampling candidates, such as `GridSearchCV` or `RandomizedSearchCV`;
- a cross-validation scheme such as `StratifiedKFold`; and
- a score function such as the defaults `sklearn.metrics.accuracy_score` for classification and `sklearn.metrics.r2_score` for regression.

The main methods are:
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#gridsearchcv) that performs an exhaustive search over specified parameter values for an estimator;
- [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). In contrast to `GridSearchCV`, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions.
- Scikit-learn also provides the `HalvingGridSearchCV` and `HalvingRandomSearchCV` estimators that can be used to search a parameter space using successive halving.

**Script** to apply a randomized search over a random forest classifier for the Iris data set: (https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_randomizedsearchCV.ipynb)

---

</details>

<details markdown="block">
<summary>Looking at different performance evaluation metrics</summary>

## Looking at different performance evaluation metrics

### Confusion matrix

The confusion matrix, also called error matrix, is a very useful tool to evaluate the precision of a classifier.

To compute the error matrix for a classifier ${\bf f_{\bf w}}({\bf x})$ trained with a given training set of examples, the steps are the following.

1. Consider a test set of examples $({\bf x}, y)$ that were not used for training;

2. Predict the labels $\hat{y}={\bf f_{\bf w}}({\bf x})$ for all examples in the test set;

3. Compare the predicted labels $\hat{y}$ with the true labels $y$ and create a two-way table where the rows represent the actual labels $y$  and the columns represent the predicted labels $\hat{y}$.

The following code illustrated how to compute a confusion matrix for a classification task with two classes, labeled 0 and 1, and plot the result with `matplotlib`. The matrix compares the true labels of the examples `y_true` with the labels predicted by the classifier `y_pred`:

<details markdown="block">
<summary>Script to compute confusion matrix given actual and predicted values</summary>

  ```
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Data
  y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
  y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
  
  # Plot confusion matrix
  ConfusionMatrixDisplay.from_predictions(
      y_true, 
      y_pred, 
      display_labels=['Zero', 'One'],
      cmap='Blues',
      colorbar=True
  )
  plt.title('Confusion Matrix')  # Optional title
  plt.tight_layout()
  plt.show()
  ```

</details>

### Accuracy metrics derived from the confusion matrix

In general, if there are $n$ different label values, the error matrix is $n \times n$. For simplicity, let's just consider the $2 \times 2$ error matrix, where correct predictions are called TP or TN, and the errors FP or FN.

|           | Predicted Positive | Predicted Negative |
|-----------|--------------------|--------------------|
| Actual Positive | TP=True Positive     | FN=False Negative    |
| Actual Negative | FP=False Positive| TN=True Negative|

<details markdown="block">
<summary>Metrics are computed from the error matrix</summary>

1. Classification **accuracy**.

$${\rm accuracy}=\frac{{\rm TP}+{\rm TN}}{{\rm TP}+{\rm FN}+{\rm FP}+{\rm TN}}.$$

If the number of actual positive examples (TP+FN) is very different from the number of negative examples (FP+TN), the largest number is going to dominate the result. For instance, is 5% of some area is burned, but the classifier just labels all pixels as non-burned, the classification accuracy will be 95%.

For that example, the error matrix will look something that the following one.


|           | Predicted Burned | Predicted Non burned |
|-----------|--------------------|--------------------|
| Actual Burned | TP=0   | FN=50    |
| Actual Non burned | FP=0| TN=9050|

---

2. **Precision**, focused on predicted positives

$${\rm precision}=\frac{{\rm TP}}{{\rm TP}+{\rm FP}}.$$

This metric focusses only on the positive examples. For the burned area example above, the precision is not defined since no predictions are positive. Consider this other example, where one aims af finding greenhouses in a certain region.


|           | Predicted Greenhouse | Predicted Other |
|-----------|--------------------|--------------------|
| Actual Greenhouse | TP=80   | FN=20    |
| Actual Other  | FP=10| TN=9090|

In that case, precision is $80/(80+10) \approx 89\%$, while overall classification accuracy is $91.7\%$.

Precision is the complement of **commission error**:

$${\rm CE}=\frac{{\rm FP}}{{\rm TP}+{\rm FP}}.$$

---

3. **Recall**, focused on actual positives, and also called **sensitivity** or **true positive rate (TPR)**

$${\rm recall}=\frac{{\rm TP}}{{\rm TP}+{\rm FN}}.$$

The denominator here is the total number of actual positives. This is an interesting metric if we are focused on having a very low error on missing an actual positive (a typical example is missing a tumor in medecine).

For the burned area example, the classifier has the worst possible outcome since it misses all actual positives, and therefore its ${\rm recall}=0\%$. However, a similarly arbitrary classifier that would just predict the *positive* label for all examples would have a perfect ${\rm recall}=100\%$. For the greenhouse example, we have ${\rm recall}=80\%$.

Recall is the complement of **omission error**:

$${\rm OE}=\frac{{\rm FN}}{{\rm TP}+{\rm FN}}.$$

For instance, one wants the *sensitivity* of a disease test to be high to ensure that sick people are detected.

---

4. **Specificity**, is focused on actual negatives, and is also called **true negative rate (TNR)**

$${\rm specificity}=\frac{{\rm TN}}{{\rm TN}+{\rm FP}}.$$

For instance, one wants the *specificity* of a disease test to be high to prevent healthy people from being labeled as sick.

---

5. **F1 score**, which averages equally *precision* and *recall*

$${\rm F1~score}= 2 \times \frac{{\rm precision} \times {\rm recall}}{{\rm precision} + {\rm recall}}=\frac{{\rm 2\\, TP}}{{\rm 2\\, TP}+{\rm FP}+{\rm FN}}.$$

This is also known as the **Dice coefficient**. For the burned area example ${\rm F1~score}=0$ since in fact the F1 score is the *harmonic mean* of precision and recall. This metric still does not take into consideration true negatives (TN) and is criticized for giving the same importance to precision and recall.

---

</details>

### The classification report 

`sklearn` offers a function that outputs a *classification report* that includes precision, recall and F1 score, for both possible labelings of the examples:

  ```
  from sklearn.metrics import classification_report
  import numpy as np
  # Actual labels
  y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
  # Predicted labels
  y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
  # Compute confusion matrix
  report = classification_report(y_true, y_pred)
  print(report)
  ```

</details>

