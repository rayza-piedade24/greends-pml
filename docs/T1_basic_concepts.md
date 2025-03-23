# Basic concepts of supervised machine learning

In this course we are dealing with data sets of *labeled examples*. Examples can be described by scalar numbers, rows of tabular data, images, etc. For tabular data, we refer the to columns as *explanatory variables* (sometimes also called *independent* or *descriptive* variables).

Labels can be categorical, ordinal or continuous. Labels can be refered to as the *response variable* (or *dependent* variable). They are also called *targets*. Typically, we the problems are called:
1. *Regression problems*, when the labels are continuous.
2. *Classification problems*, when the labels are categorical.

The distinction is not always clear. Some problems can be considered either as regression or classification problems.

Given a supervised ML problem, i.e. a set of labeled examples, the goal is to build a function $f$ that maps examples to labels or, in other words, that *predicts* the label from the example.

The outputs of $f$ are called *predictions* or *predicted values*, and the actual labels of the examples are called *actual values* or *target values*.

---

<details markdown="block">
<summary>  Models and parameters </summary>

## Models and parameters

More formally, if $E$ is the set of examples and ${\rm labels}$ is a set that includes the labels, then what we call the *model* is a family of functions $f_{\rm \bf w}$ that depends on a set of parameters ${\rm \bf w}$ that maps an example into a label: $$f_{\rm \bf w}: E → {\rm labels}.$$

It can be more convenient to express the function as depending on the parameters ${\rm \bf w}$ as well as the example ${\rm \bf x}$. The model's predicted label $\hat{y}$ for the example ${\rm \bf x}$ is:

$$\hat{y}=f_{\rm \bf w}({\rm \bf x})= f({\rm \bf x}; {\rm \bf w}).$$

ML practicioners use an enormous variety of models, depending on the problem at hand and on the available computational resources to train the model. Models include convolucional neural networks (CNN) for image classification (resnet and other kind of CNNs), neural networks (NN) for classification of tabular data, linear regression models, decision and regression trees, random forest and other ensemble models, among many other models.


<img src="https://drive.google.com/uc?export=view&id=1g7cUxaqoa1ujkRV-BjQKPWsVDFBAYWMb" width="600" >



### Example of a simple model (simple linear regression)

Suppose that our examples are scalar numbers $x_1,\dots, x_n$ and the labels are continuous labels $y_1, \dots, y_n$. We call $x$ the explanatory variable and $y$ the response variable.

Let's consider the simple linear regression model:
$f_{\rm w_0,w_1}(x)= w_0 + w_1 \\, x$. The model parameters are ${\rm \bf w}=(w_0,w_1)$ and the predicted values are given by 
$$\hat{y}= w_0 + w_1 \\, x.$$

The target  or actual label values are the $y_1, \dots, y_n$, and the predicted label values are 
$$\hat{y}\_1=f_{\rm w_0,w_1}(x_1),\dots,\hat{y}\_n=f_{\rm w_0,w_1}(x_n).$$

### Example of a simple model (quadratic regression)

This case is similar to the previous except the model has an additional weight associated to a quadratic term since the model $f_{\rm a,b,c}$ in that example is quadratic instead of linear. This gives increased flexibility to the model. In general the linear model can be extended to a polynomial model of any degree, with the addition of more parameters. This increases flexibility but also increases the risk of overfitting.

$$f_{\rm w_0,w_1,w_2}(x)= w_0 + w_1 \\, x + w_2 \\, x^2.$$

### Multiple linear regression

In most practical cases, there are more than one explanatory variable. For instance, for the Iris data set, we could consider that the explanatory variables are `SepalWidth` ($x_1$),  `PetalLength` ($x_2$) and `PetalWidth` ($x_3$) and the response variable is `SepalLength`  ($y$). The linear regression model is 
$$\hat{y}=w_0 + w_1 \\, x_1 +  w_2 \\, x_2 + w_3 \\, x_3.$$

<!---
Note that now $x_1$ represents one explanatory variable. To represent all observations, we typically use a bold notation. So, ${\rm \bf x}_1$ represents all $n$ observations of the variable $x_1$ for the $n$ examples in the data set ($n=150$ for the complete `Iris`data set). In that case the $n$ observations are ${\rm \bf x}\_1$, ${\rm \bf x}\_2$ and ${\rm \bf x}\_3$, and the labels for the $n$ observations are  `SepalLength`  (${\rm \bf y}$), where each symbol at bold represents a *vector* of observations
-->

### Pseudo-code for linear regression

The following pseudo-code describes a sequence of steps find a good solution for the multiple linear regression problem. We start by reading the $n$ observations. The *hyperparameters* are the *learning rate* and the *number of iterations*. In each iteration the *weights* are updated according to the *error*, i.e. the difference between predicted and actual values of the response variable. The goal of the algorithm is to iteratively reduce the errors by converging to a better set of weights.

---
  1. Dataset:  $D = {(x_1^{(i)}, ..., x_n^{(i)}, y^{(i)})}\_{i=1}\^n$  `n example, p features`
  2. Learning rate:  $\eta$ `Small positive value`
  3. Max iterations: max_iter `Number of epochs`
  4. Initial weights $w:=(w_0, w_1, ...,w_p)$ `Typically, all zero`
  5. For ${\rm iter}:=1$ to max_iter: 
     - For each  $(x_1, ..., x_p, y) \in D$  `Update weights after each example`
       - $\hat{y}:=w_0 + w_1 \\, x_1 + w_2 \\, x_2 + \dots + w_n \\, x_p$ `Predict response with current weights for the LR model`
       - error:= $y-\hat{y}$
       - $w_0:=w_0 + \eta \cdot {\rm error}$ # `Update weight (bias)`
       - For $j:=1$ to $p$
         - $w_j:=w_j +\eta \cdot {\rm error} \cdot x_j$ # `Update weight (for each feature)`
---      

</details>
<details markdown="block">
<summary> Loss function for regression </summary>

## Loss function for regression

In supervised ML, it is usual to call *loss* to the **dissimilarity** between actual and predicted label values for a *set* of labeled examples.

Let ${\rm \bf x}\_1, \dots , {\rm \bf x}\_n$ be a set of examples (note that now the index $1,\dots,n$ refers to the example and not to the explanatory variable) with labels $y_1, \dots , y_n$. Let $f_{\rm \bf w}$ be our model. Therefore, the predicted labels are

$$\hat{y}\_1=f_{\rm \bf w}({\rm \bf x}\_1), \dots, \hat{y}\_n=f_{\rm \bf w}({\rm \bf x}\_n).$$

The loss over that set of examples is some dissimilarity measure between the actual labels $y_1, \dots , y_n$ and the predicted labels $\hat{y}\_1, \dots , \hat{y}\_n$.

### Dissimilarity measures to define *loss*

To define loss, we then need to choose an appropriate dissimilarity metric between a set of actual $y_1, \dots , y_n$ and predicted labels $\hat{y}\_1, \dots , \hat{y}\_n$. The choice depends on the type of problem, and while MAE or MSE are adequate for *regression* problems, other dissimilarities are used for *classification* problems.

For a set of $n$ examples, the following expressions are commonly used to define the loss for a regression problem:

1. Mean absolute error (MAE), given by $\frac{1}{n}\sum_{i=1}^n |y_i-\hat{y}_i|$; or

2. Mean square error (MSE), given by $\frac{1}{n}\sum_{i=1}^n \left(y_i-\hat{y}_i\right)^2$

In the one hand, MAE is not differentiable everywhere, which is an undesirable property for ML. On the other hand, MSE penalizes too much large differences between actual and predicted values, which means that a single example can constraint strongly the solution.

An alternative is called the Huber loss function, which is differentiable everywhere, and behaves like MSE near the origin and like MAE for large $|y_i-\hat{y}_i|$.

---

</details>
<details markdown="block">
<summary> ML as an optimization problem </summary>
  
## ML as an optimization problem

Now, we can define a ML problem as a optimization problem. Given

1.  a set of examples  ${\rm \bf x}_1, \dots , {\rm \bf x}_n$  with labels $y_1, \dots , y_n$
2. a model $f_{\rm \bf w}$
3. a *loss* function $L$

The goal is to determine the optimal set of parameters ${\rm \bf w}$ that minimize the loss $L$ over that set of examples. Next, we discuss how in practice ML methods find a solution (the best set of weights) for this problem.

### Gradient descent and learning rate

Informally, a gradient measures how much the output of a function changes if you change the inputs a little bit.

Given a model $f_{\rm \bf w}({\rm \bf x})= f({\rm \bf x}; {\rm \bf w})$ and a batch of examples ${\rm \bf x_1}, \dots, {\rm \bf x_n}$, we have seen how we can define a *loss* function

$$L({\rm \bf x_1, \dots, x_n; w})= L_{\rm \bf x_1, \dots, \rm \bf x_n}(\rm \bf w).$$

We can write $L$ just a function of the weights since the ${\rm \bf x_i}$ are fixed for given batch of examples. Our goal is to find the set of weights ${\rm \bf w}$ that minimize $L({\rm \bf w})$. In order to do this iteratively, starting with an arbitrary set of initial weights, we would like to know how $L$ changes with a small change in the weights $\rm \bf w$ from the current set weights ${\rm \bf w}^{\star}$.

This  is given by the gradient of $L$ with respect to ${\rm \bf w}$ at ${\rm \bf w}^{\star}$, which is a vector of partial derivatives of $L$ with length equal to $m$=number of model parameters.

$$\nabla L({\rm \bf w}^{\star}) = \frac{\partial L}{\partial \rm \bf w}({\rm \bf w}^{\star})= \left(\frac{\partial L}{\partial \rm w_1}({\rm \bf w}^{\star}), \dots,  \frac{\partial L}{\partial \rm w_m}({\rm \bf w}^{\star}) \right).$$

The computation of $\nabla L({\rm \bf w}^{\star})$ is usually done by **back-propagation**, which is an automatic differentiation algorithm for calculating gradients for the weights in a computational graph. Back-propagation (aka *backprop*) is an automatic differentiation algorithm that applies the *chain-rule*.

The vector $\nabla L({\rm \bf w}^{\star})$ points to the direction from ${\rm \bf w}^{\star}$ along which $L$ grows faster, so gradient descent follows the opposite direction $-\nabla L({\rm \bf w}^{\star})$.

<img src="https://drive.google.com/uc?export=view&id=1-KGjbUaR1l3z879V_eJu7JutnSutqbdC" width="500" >


To simplify, let's suppose that all examples are visited before updating the set of weights.
Then, the steps of gradient descent algorithm are the following. In ML, one *epoch* corresponds to the processing of the totally of examples in the data set. So, for instance, if the algorithm runs for 20 epochs, then the model is applied to all examples 20 times.

---

1. Choose an initial set of weights ${\rm \bf w}^{\star}$

2. For $i = 1, \dots, E$, where $E$ is the number of epochs, do:

   i) Cumpute $\nabla L({\rm \bf w}^{\star})$

   ii) Update ${\rm \bf w}^{\star}:={\rm \bf w}^{\star} - \eta\\, \nabla L({\rm \bf w}^{\star})$, where $\eta>0$ is the learning rate.

---

The choice of the *learning rate* is critical for a good performance of the algorithm. A very small learning rate will permit a good approximation of the gradient flow by the algorithm (see next figure). But if the step is too small, many epochs will be needed to get a good solution.

<img src="https://drive.google.com/uc?export=view&id=12c4X3po4-xVGUJKzyKC56lwl4ZEmXqWa" width="400" >

### Stocastich gradient descent for Linear Regression

Earlier, we looked at a pseudo-code to solve the multiple linear regression problem iteratively. The weight updates were done with the following steps:

- $w_0$ := $w_0 + \eta \cdot$ error # `Update weight (bias)`
- For $j$ := 1 to $n$
  - $w_j$ := $w_j + \eta \cdot {\rm error} \cdot x_j$ # `Update weight (for each feature)`

where the errors are given by $y-\hat{y}$ and $\eta$ is the learning rate. What has this to do with the loss function and the gradient?

In fact, for the MSE loss function $L=\frac{1}{n}\sum_{i=1}^n \left(y_i-\hat{y}_i\right)^2$, it is easy to show that

- $\frac{\partial L}{\partial w_0}=-2 \\, {\rm error}$, and 
- $\frac{\partial L}{\partial w_j}=-2 \\, {\rm error}\cdot x_j$ , for any other model weight ($j > 0$)

Since those expressions correspond  to the updates in the pseudo-code, this shows that the pseudo-code is in fact using the MSE loss function and *gradient descent* to update the weights, with $\eta$ as the learning rate. The algorithm is called *stochastic* because the weights are updated after each example is assessed. The alternative is to use *batches of examples* and update weights once per batch. The extreme case of batch processing is to have a single batch containing all examples. In such case the weights are updated only once per epoch.  

---

</details>
<details markdown="block">
<summary> Computing gradients with PyTorch </summary>

## Computing gradients with PyTorch

Video suggestion: [Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4). The author explains what is a computation graph and how PyTorch uses it to compute gradients. The example uses a very simple model: $\hat{y}=w \cdot x$ and the MSE loss which is just $(\hat{y}-y)\^2$.

- Pipeline for the regression problem:
  - Prepare data
  - Design model (input, output size, model: perceptron)
  - Construct loss and optimizer
  - Training loop
    - Forward pass: prediction and loss
    - Backward pass: gradients
    - Update weights

In the following examples, one starts with a step by step code in Python for the linear regression model  that is based on our knowledge of  the gradient expression for the MSE loss function and we convert it into a Pytorch code that can be easily generalized to other models and other loss functions.

Video suggestion: [Gradient Descent with Autograd and Backpropagation by Patrick Loeber](https://www.youtube.com/watch?v=E-I2DNVzQLg). The author first uses `numpy` to create a gradient descent script for a  linear regression model and then replaces the manual gradient calculation by `PyTorch` automatic gradient calculation. The following scripts illustrate the possibilities to move from a low level Python code to a higher level PyTorch code for the simple Linear Regression problem. The higher-level code can be easily adapted to more complex models and other optimizers.

- [numpy version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_1_gradientdescent_manually.py)
- [torch version](https://github.com/patrickloeber/pytorchTutorial/blob/master/05_2_gradientdescent_auto.py)
- [torch version with torch loss criterion and optimizer](https://github.com/patrickloeber/pytorchTutorial/blob/master/07_linear_regression.py)

The following table illustrates the changes from a basic Python script which is dependent on the model, loss, etc,  to a PyTorch higher-level script that can easily generalized to other models, loss functions and optimizer strategies.

| Basic Python | PyTorch 
|---|---
| Define model explicitly | Use a pre-defined model
|`def predict(x):`|`torch.nn.Linear(in_size,out_size)`
| Define loss explicitly | Use a pre-defined loss function
|`def loss(y,y_pred):`|`loss=torch.nn.MSEloss(y,y_pred)`
| Loss optimization strategy | Use a pre-defined optimizer
| | `optimizer=torch.optim.SGD(params, learn_rate)`
| Compute *ad hoc* gradient | **Use built-in backpropagation mechanism**
|`def gradient(x,y,y_pred):`|`loss.backward()`
|Update weights explicitly| `optimizer.step()`

---

</details>
<details markdown="block">
<summary> Exercise with pseudo-code for SGD </summary>

## Exercise with pseudo-code for SGD

Consider the following pseudo-code to train a simple Linear Regression model. What is the *loss* function that we aim at minimizing? What is the strategy to reduce the *loss* in each iteration? Is there a risk of *over-fitting*?
  
  ---
  Pseudo code for SGD (stochastic gradient descent) to fit a linear regression:
  
  - Dataset:  $D = {(x_1^{(i)}, ..., x_n^{(i)}, y^{(i)})}\_{i=1}\^N$  `N observations, n features`
  - Learning rate:  $\eta$ `Small positive value`
  - Max iterations: max_iter `Number of epochs`
  - Initial weights $w$ := $(w_0, w_1, ..., w_n)$ `Typically, all zero`
  - For iter := 1 to max_iter 
    - For each  $(x_1, ..., x_n, y) \in D$  `Update weights after each example`
      - $\hat{y}$ := $w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$ `Predict response with current weights`
      - error := $y-\hat{y}$
      - $w_0$ := $w_0 + \eta \cdot$ error # `Update weight (bias)`
      - For $j$ := 1 to $n$
        - $w_j$ := $w_j + \eta \cdot$ error $\cdot x_j$ # `Update weight (for each feature)`
          
  ---

- Create a `LinearRegression` class with a `fit` method to implement the pseudo code above. Add to your class a `predict` method to make new predictions using the fitted model. Test your class with the following example.
    
  ```Python
  # Create synthetic data
  np.random.seed(0)
  X = np.random.rand(100, 1) # array with 100 rows and 1 column (1 feature)
  y = 2 + 3 * X + np.random.randn(100, 1) * 0.1
  # Create and train the model
  model = LinearRegression(learning_rate=0.1, max_iter=1000)
  model.fit(X, y)
  # Make predictions
  X_test = np.array([[0.5]])
  y_pred = model.predict(X_test)
  print(f"Prediction for X=0.5: {y_pred[0]}")
  ```
- Create an animation that shows the position of the fitted line for successive epochs for the example above.
- How can you adapt the code to address a classification problem where the response $y$ can only be 0 or 1?

</details>

<!---

Below, we discuss a `PyTroch` gradient descent script for the linear regression problem, and we compare the result with the optimal coefficients obtained by *least squares*. The code below shows how *training loss* is  computed.

The most specific part of the algorithm is the gradient computation. Note that the *gradient machinery* of `PyTorch` is turned-on for each weight with `requires_grad = True` as in the following case:

    coeffs=torch.tensor([-20.,-10.]).requires_grad_()

Then, the derivatives can be computed for any continuous function of the weights in tensor `coeffs`. In particular, the *loss* $L$ is defined as a function (that can be arbitrarily complicated) of the weights, and the *gradient* $\nabla L({\rm \bf w}^{\star})$ for the current set of weights ${\rm \bf w}^{\star}$ is computed with

    loss.backward()

Finally, the weights are updated with

    coeffs.sub_(coeffs.grad * step_size)

where method `sub_` is substraction for weight updating ${\rm \bf w}^{\star}:={\rm \bf w}^{\star} - \eta \\, \nabla L({\rm \bf w}^{\star})$, and  the learning rate $\eta$ is called `step_size` in the code.

PyTorch accumulates the gradients on subsequent backward passes, i.e. it accumulates the gradients on every `loss.backward()` call. Since  the update is to be based on the current gradient value, we need to include the `coeffs.grad.zero_()` instruction to zero the gradients before the next pass.

Try changing the learning rate to see what is the result (try for instance `step_size=0.1`).

<details>
  <summary>Script: gradient descent with PyTorch, train only, stochastic gradient descent</summary>

```python
# This example illustrates: gradient descent with PyTorch, train only, stochastic gradient descent (SGD)
import matplotlib.pyplot as plt
import torch
import numpy as np
torch.manual_seed(42)

step_size = 0.001  # learning rate
iter = 20  # number epochs

############################################ Creating synthetic data
# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)  # view converts to rank-2 tensor with one column
func = -5 * X + 2
# Adding Gaussian noise to the function f(X) and saving it in Y
y = func + 0.4 * torch.randn(X.size())

########################################## Baseline: Linear regression LS solution
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
print('Least square LR coefficients:',reg.intercept_,reg.coef_)

####################################################### Gradient Descent
# initial weights
coeffs = torch.tensor([-20., -10.], requires_grad=True)

# defining the function for prediction (linear regression)
def calc_preds(x):
    return coeffs[0] + coeffs[1] * x

# Computing MSE loss for one example
def calc_loss_from_labels(y_pred, y):
    return torch.mean((y_pred - y) ** 2) # MSE

# lists to store losses for each epoch
training_losses = []

# epochs
for i in range(iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = calc_preds(X)
    loss = calc_loss_from_labels(y_pred, y)
    training_losses.append(loss.item())

    # Stochastic Gradient Descent (SGD): update weights 
    for j in range(X.shape[0]):
        # randomly select a data point
        idx = np.random.randint(X.shape[0])
        x_point = X[idx]
        y_point = y[idx]

        # making a prediction in forward pass
        y_pred = calc_preds(x_point)

        # calculating the loss between predicted and actual values
        loss = calc_loss_from_labels(y_pred, y_point)

        # compute gradient
        loss.backward()

        # update coeffs
        with torch.no_grad():
            coeffs.sub_(coeffs.grad * step_size)
            # zero gradients
            coeffs.grad.zero_() # prevents from accumulating

print('coeffs found by stochastic gradient descent:', coeffs.detach().numpy())

# plot training loss along epochs
plt.plot(training_losses, '-g')
plt.xlabel('epoch')
plt.ylabel('loss (MSE)')
plt.show()
```
</details>

--->
