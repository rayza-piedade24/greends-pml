import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# remove features (optional)
# df = df[['petal width (cm)','petal length (cm)','sepal width (cm)','sepal length (cm)']]
feature_names = np.array(df.columns)
X = df.values  # or df.to_numpy()
y = iris.target

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Create Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X, y)

# Create AdaBoost classifier with decision tree base estimator
ada_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), random_state=42)
ada_clf.fit(X_train, y_train)

# Create Gradient Boosting classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)

# Compute permutation importance
rf_result = permutation_importance(rf_clf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
ada_result = permutation_importance(ada_clf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
gb_result = permutation_importance(gb_clf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Use the mean importances
rf_importance = rf_result.importances_mean
ada_importance = ada_result.importances_mean
gb_importance = gb_result.importances_mean

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
ax.set_title('Permutation Importance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
