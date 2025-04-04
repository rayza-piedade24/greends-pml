# Ensemble methods

<details markdown="block">
<summary> Random Forests </summary>

## Random forests

Recall decision and regression trees -- see for instance [Normalized Nerd videos](https://www.youtube.com/channel/UC7Fs-Fdpe0I8GYg3lboEuXw) on classification and regression trees.

Random forests are ensemble learning methods that involve:
  - (bootstraping) Creating a collection of trees from bootstrap samples (sampling with replacement); [see meaning](https://en.wikipedia.org/wiki/Bootstrapping)
  - (decorrelating) Decorrelate models by randomly selecting features
  - (aggregating) Ensembling the collection of trees by majority vote.

<img src="https://github.com/isa-ulisboa/greends-pml/blob/main/figures/random_forests.png" width="600" >

The pseudo-code below describes the main steps to create a random forest.

---

  <details markdown="block">
  <summary> Pseudo-code to create and apply a Random Forest</summary>
  
  ### Step 1: Initialize Parameters
  1. Set the number of trees `N_trees`.
  2. Define the maximum depth of each tree `max_depth`.
  3. Set the number of features to consider when splitting `max_features`.
  
  ### Step 2: Prepare the Data
  1. Split the dataset into training and testing sets.
  2. Preprocess the data (e.g., handle missing values, normalize if needed).
  
  ### Step 3: Build the Random Forest
  1. Initialize an empty list `forest` to store decision trees.
  
  2. For each tree `i` in range(1, N_trees):
     - **Step 3.1:** Create a bootstrap sample:
       - Randomly sample the training data with replacement to create a subset.
     - **Step 3.2:** Train a decision tree:
       - Select a random subset of features (`max_features`).
       - Grow the tree using the bootstrap sample:
         - At each node, split on the best feature (based on criteria like Gini Impurity or Entropy for classification, or variance for regression).
         - Stop splitting if `max_depth` is reached or other stopping criteria are met.
     - **Step 3.3:** Add the trained decision tree to `forest`.
  
  ### Step 4: Make Predictions
  1. For a new data point:
     - Pass it through each tree in the forest.
     - Collect predictions from all trees (majority vote for classification, or weighted mean for regression).
  
  2. Return the final prediction.
  
  ### Step 5: Evaluate Performance
  1. Use the testing set to evaluate accuracy or other metrics (e.g., precision, recall).
  
  ---
  
  </details>
  
  <details markdown="block">
  <summary> Why do random forest reduce the variance of the estimator?</summary>
  
  
  For simplicity, let's consider regression trees and show that the goal of ensembling trees with random forests is reducing the variance. 
  
  Let  $X_i$ be  the random variable  that represents the predition for the regression tree $T_i$ from the collection, with $\rho={\rm cor}[X_i,X_j]$ being the correlation between $X_i$ and $X_j$. The prediction from the ensemble is
  
  $$\bar{X}=\frac{1}{n} \left( X_1+\dots+X_B \right)$$
  
  and its variance is given by
  
  $${\rm Var}[\bar{X}]=  \rho \, \sigma^2 + \frac{1-\rho}{B} \sigma^2,$$
  
  where ${\rm Var}[X-i]=\sigma^2$ and $B$ is the number of bootstrap samples. As long as $\rho$ does not grow with $B$, which is why the trees are decorrelated, using a larger ensemble will increase $B$ and reduce ${\rm Var}[\bar{X}]$, which is the goal of ensembling estimators.
  
  ---
  
  </details>
  
  <details markdown="block">
  <summary> Script to create random forest with scikit-learn</summary>
  
  ```
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  
  # Load the Iris dataset
  iris = load_iris()
  X = iris.data
  y = iris.target
  
  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Create a Random Forest classifier
  rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
  # Train the classifier
  rf_classifier.fit(X_train, y_train)
  # Make predictions on the test set
  y_pred = rf_classifier.predict(X_test)
  # Evaluate the accuracy of the classifier
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  ```
  
  
  </details>
  
  ---

</details>

<details markdown="block">
<summary> Adaboost </summary>

## Adaboost

As discussed in [(Sagi and Rokach, 2017)](docs/Sagi_2018_Ensemble_learning_A_survey_Wire.pdf), the
main idea of AdaBoost is to focus on instances that were previously misclassified when training a new inducer. The
level of focus given is determined by a **weight** that is assigned to each instance in the training set. In the first iteration,
the same weight is assigned to all of the instances. In each iteration, the weights of misclassified instances are increased,
while the weights of correctly classified instances are decreased. In addition, weights are also assigned to the individual
base learners based on their overall predictive performance.

**AdaBoost** is a *dependent* ML method since each tree is an improvement over previous trees in the sequence. This is the opposite of *random forests* where the tree are grown independently.

--- 

</details>

<details markdown="block">
<summary> Gradient boosting </summary>

## Gradient boosting

Gradient Boost is also a *dependent* method, since each tree is an improvement of the earlier trees. Gradient Boost provides a framework to build an ensemble of trees based on an arbitrary loss function. In Gradient Boosting, each new tree is computed using a **simple classifier** (also called weak inducer, that just performs better than random) the **residuals** from the previous model.

For details and very nice illustrations, look at the two following posts:

1. [Regression](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502)

2. [Classification](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e)

--- 

</details>

<details markdown="block">
<summary> Variable importance</summary>

## Variable importance

Since interpretability is a concept difficult to define precisely, people eager to gain
insights about the driving forces at work behind random forests predictions often focus
on variable importance, a measure of the influence of each input variable to predict
the output [Scornet, 2021](https://arxiv.org/pdf/2001.04295). In Breiman (2001) original random forests, there exist two importance
measures:

1. **Mean Decrease Impurity**, MDI, or Gini importance, see Breiman (2002),
which sums up the gain associated to all splits performed along a given variable; and

2. **Mean Decrease Accuracy**, MDA, or **permutation importance**, see Breiman (2001), 
which shuffles entries of a specific variable in the test data set and computes the
difference between the error on the permuted test set and the original test set.

Because
of its very definition, MDI is an importance measure that can be computed for trees
only, since it strongly relies on the tree structure, whereas MDA is an instantiation of
the permutation importance that can be used for any predictive model. Both measures
are used in practice even if they possess several major drawbacks.

**MDI** is known to favor variables with many categories. Even when variables have the same number of categories,
MDI exhibits empirical bias towards variables that possess a category having a high frequency. MDI is also biased in presence of correlated features.

**MDA** seems to exhibit less bias than MDI but tends to overestimate correlated features. 


<details markdown="block">
<summary> Script to compute MDI for different classifiers for the Iris data set</summary>

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X, y)

# Create AdaBoost classifier with decision tree base estimator
ada_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), random_state=42)
ada_clf.fit(X, y)

# Create Gradient Boosting classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X, y)

# Calculate feature importance for each classifier
rf_importance = rf_clf.feature_importances_
ada_importance = ada_clf.feature_importances_
gb_importance = gb_clf.feature_importances_

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the feature importance
x = np.arange(len(feature_names))
width = 0.2

rects1 = ax.bar(x - width, rf_importance, width, label='Random Forest')
rects2 = ax.bar(x, ada_importance, width, label='AdaBoost')
rects3 = ax.bar(x + width, gb_importance, width, label='Gradient Boosting')

# Add labels, title, and legend
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
```
---

</details>
