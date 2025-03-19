- [sklearn Estimators that handle NaN values](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)

- [sklearn missing values support](https://scikit-learn.org/stable/modules/tree.html#missing-values-support)

<details markdown="block">
<summary> Example with the Iris data set </summary>

```
from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
iris = load_iris()

def convert_to_nan(array, percentage=0):
    """
    Randomly convert a percentage of entries in a 2D array to np.nan.
    
    Parameters:
    array (numpy.ndarray): Input 2D array
    percentage (float): Percentage of entries to convert to NaN (default: 0.1 for 10%)
    
    Returns:
    numpy.ndarray: Array with randomly selected entries converted to NaN
    """
    # Ensure the input is a numpy array
    array = np.array(array)
    
    # Calculate the number of elements to convert to NaN
    num_nan = int(percentage * array.size)
    
    # Generate random indices for NaN placement
    nan_indices = np.random.choice(array.size, num_nan, replace=False)
    
    # Create a copy of the array, flatten it, and set the selected indices to NaN
    result = array.copy().flatten()
    result[nan_indices] = np.nan
    
    # Reshape the array back to its original shape
    return result.reshape(array.shape)

X = convert_to_nan(iris.data)
print(X)
y = iris.target
print(' labels: ', iris.target_names)

#build decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4,min_samples_leaf=4,splitter='best')
#max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node

#fit the tree to iris dataset
clf.fit(X,y)

#plot decision tree
fig, ax = plt.subplots(figsize=(10, 10)) #figsize value changes the size of plot
tree.plot_tree(clf,ax=ax,feature_names=['sepal length','sepal width','petal length','petal width'])
plt.show()
```

</details>
