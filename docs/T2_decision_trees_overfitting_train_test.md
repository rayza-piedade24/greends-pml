So far all ''from scratch'' examples above (simple linear regression, quadratic regression) dealt with *scalar* inputs, i.e. each example was described by a single number.

Examples can also be tabular data, where each example is described by a numeric vector. Formally, the $i$-th example is described by a vector  $(x_{i1}, \dots, x_{ik})$ of length $k$, for examples $i = 1, \dots, n$ and labels are $y_1, \dots,  y_n$  as before.

# Decision trees

The example below shows a decision tree for the iris data set. The root node represents the 4-dimensional space defined by the variables sepal length,sepal width, petal length, petal width. This is a 3-class problem where labels are the varieties setosa, versicolor, virginica.

We call depth of the decision tree to the maximum number of splits to define a leaf node. Note that the code establishes max_depth=4 to prevent the tree from growing more than 4 levels. The figure indicates the number of examples (or training samples) that lie in each node of the tree.



<details markdown="block">
<summary> Basic decision tree classifier, plot tree (sklearn) </summary>

    from sklearn.datasets import load_iris
    from sklearn import tree
    from matplotlib import pyplot as plt
    iris = load_iris()
    
    X = iris.data
    y = iris.target
    print(' labels: ', iris.target_names)
    
    #build decision tree
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4,min_samples_leaf=4)
    #max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node
    
    #fit the tree to iris dataset
    clf.fit(X,y)
    
    #plot decision tree
    fig, ax = plt.subplots(figsize=(10, 10)) #figsize value changes the size of plot
    tree.plot_tree(clf,ax=ax,feature_names=['sepal length','sepal width','petal length','petal width'])
    plt.show()
</details>


