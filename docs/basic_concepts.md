# Models and parameters


More formally, if $E$ is the set of examples and $L$ is a set that includes the labels, then what we call the *model* is a family of functions $f_{\rm \bf w}$ that depends on a set of parameters ${\rm \bf w}$: $$f_{\rm \bf w}: E → L.$$

It can be more convenient to express the function as depending on the parameters ${\rm \bf w}$ as well as the example ${\rm \bf x}$. The model's predicted label $\hat{y}$ for the example ${\rm \bf x}$ is:

$$\hat{y}=f_{\rm \bf w}({\rm \bf x})= f({\rm \bf x}; {\rm \bf w}).$$

ML practicioners use an enormous variety of models, depending on the problem at hand and on the available computational resources to train the model. Models include convolucional neural networks (CNN) for image classification (resnet and other kind of CNNs), neural networks (NN) for classification of tabular data, linear regression models, decision and regression trees, random forest and other ensemble models, among many other models.


<img src="https://drive.google.com/uc?export=view&id=1g7cUxaqoa1ujkRV-BjQKPWsVDFBAYWMb" width="600" >



## Example of a simple model (simple linear regression)

Suppose that our examples are scalar numbers $x_1,\dots, x_n$ and the labels are continuous labels $y_1, \dots, y_n$. We call $x$ the explanatory variable and $y$ the response variable.

Let's consider the simple linear regression model:
$f_{\rm a,b}(x)= a \, x + b$. The model parameters are ${\rm \bf w}=(a,b)$ and the predicted values are given by 
$$\hat{y}=f(x; {\rm a,b})=a\, x + b.$$

The target  or actual label values are the $y_1, \dots, y_n$, and the predicted label values are the $\hat{y}_1,\dots,\hat{y}_n$.



## Example of a simple model (quadratic regression)

This case is similar to the previous except the model has an additional weight associated to a quadratic term since the model $f_{\rm a,b,c}$ in that example is quadratic instead of linear. This gives increased flexibility to the model. In general the linear model can be extended to a polynomial model of any degree, with the addition of more parameters. This increases flexibility but also increases the risk of overfitting.

$$f_{\rm a,b,c}(x)= f(x;a,b,c)= a \, x^2 + b \, x + c.$$



# Loss function for regression

In ML, it is usual to call *loss* to the **dissimilarity** between actual and predicted label values for a *set* of labeled examples.

Let ${\rm \bf x}_1, \dots , {\rm \bf x}_n$ be a set of examples with labels $y_1, \dots , y_n$. Let $f_{\rm \bf w}$ be our model. Therefore, the predicted labels are

$$\hat{y}_1=f_{\rm \bf w}({\rm \bf x}_1), \dots, \hat{y}_n=f_{\rm \bf w}({\rm \bf x}_n).$$

The loss over that set of examples is some dissimilarity measure between the actual labels $y_1, \dots , y_n$ and the predicted labels $\hat{y}_1, \dots , \hat{y}_n$.



## Dissimilarity measures to define *loss*


To define loss, we then need to choose an appropriate dissimilarity metric between a set of actual $y_1, \dots , y_n$ and predicted labels $\hat{y}_1, \dots , \hat{y}_n$. The choice depends on the type of problem, and while MAE or RMSE are adequate for *regression* problems, other dissimilarities are used for *classification* problems.




## Examples of loss functions for regression problems (MAE, MSE, Huber)



Above, two common loss functions for regression problems were listed

1. Mean absolute error (MAE), given by $\frac{1}{n}\sum_{i=1}^n |y_i-\hat{y}_i|$; or

2. Mean square error (MSE), given by $\frac{1}{n}\sum_{i=1}^n \left(y_i-\hat{y}_i\right)^2$

In the one hand, MAE is not differentiable everywhere, which is an undesirable property for ML. On the other hand, MSE penalizes too much large differences between actual and predicted values, which means that a single example can constraint strongly the solution.

An alternative is called the Huber loss function, which is differentiable everywhere, and behaves like MSE near the origin and like MAE for large $|y_i-\hat{y}_i|$.




# ML as an optimization problem

Now, we can define a ML problem as a optimization problem. Given

1.  a set of examples  ${\rm \bf x}_1, \dots , {\rm \bf x}_n$  with labels $y_1, \dots , y_n$
2. a model $f_{\rm \bf w}$
3. a *loss* function $L$

the goal is to determine the optimal set of parameters ${\rm \bf w}$ that minimize the loss $L$ over that set of examples.

## Gradient descent and learning rate

Informally, a gradient measures how much the output of a function changes if you change the inputs a little bit.

Given a model $f_{\rm \bf w}({\rm \bf x})= f({\rm \bf x}; {\rm \bf w})$ and a batch of examples ${\rm \bf x_1}, \dots, {\rm \bf x_n}$, we have seen how we can define a *loss* function

$$L({\rm \bf x_1, \dots, x_n; w})= L_{\rm \bf x_1, \dots, \rm \bf x_n}(\rm \bf w).$$

We can write $L$ just a function of the weights since the ${\rm \bf x_i}$ are fixed for given batch of examples. Our goal is to find the set of weights ${\rm \bf w}$ that minimize $L({\rm \bf w})$. In order to do this iteratively, starting with an arbitrary set of initial weights, we would like to know how $L$ changes with a small change in the weights $\rm \bf w$ from the current set weights ${\rm \bf w}^{\star}$.

This  is given by the gradient of $L$ with respect to ${\rm \bf w}$ at ${\rm \bf w}^{\star}$, which is a vector of partial derivatives of $L$ with length equal to $m$=number of model parameters.

$$\nabla L({\rm \bf w}^{\star}) = \frac{\partial L}{\partial \rm \bf w}({\rm \bf w}^{\star})= \left(\frac{\partial L}{\partial \rm w_1}({\rm \bf w}^{\star}), \dots,  \frac{\partial L}{\partial \rm w_m}({\rm \bf w}^{\star}) \right).$$

The computation of $\nabla L({\rm \bf w}^{\star})$ is usually done by **back-propagation**, which is an automatic differentiation algorithm for calculating gradients for the weights in a neural network graph structure. Back-propagation (aka *backprop*) is an automatic differentiation algorithm that applies the *chain-rule*.

The vector $\nabla L({\rm \bf w}^{\star})$ points to the direction from ${\rm \bf w}^{\star}$ along which $L$ grows faster, so gradient descent follows the opposite direction $-\nabla L({\rm \bf w}^{\star})$.

<img src="https://drive.google.com/uc?export=view&id=1-KGjbUaR1l3z879V_eJu7JutnSutqbdC" width="500" >


To simplify, let's suppose that all examples are visited before updating the set of weights.
Then, the steps of gradient descent algorithm are the following. In ML, one *epoch* corresponds to the processing of the totally of examples in the data set. So, for instance, if the algorithm runs for 20 epochs, then the model is applied to all examples 20 times.

---

1. Choose an initial set of weights ${\rm \bf w}^{\star}$

2. For $i = 1, \dots, E$, where $E$ is the number of epochs, do:

   i) Cumpute $\nabla L({\rm \bf w}^{\star})$

   ii) Update ${\rm \bf w}^{\star}:={\rm \bf w}^{\star} - \eta \, \nabla L({\rm \bf w}^{\star}) $, where $\eta >0 $ is the learning rate.

---

The choice of the *learning rate* is critical for a good performance of the algorithm. A very small learning rate will permit a good approximation of the gradient flow by the algorithm (see next figure). But if the step is too small, many epochs will be needed to get a good solution.

<img src="https://drive.google.com/uc?export=view&id=12c4X3po4-xVGUJKzyKC56lwl4ZEmXqWa" width="400" >


Let's consider a very simple example, where we try to fit a model to a pairs of observation that are linearly related. Below, we discuss a `PyTroch` gradient descent script for the linear regression problem, and we compare the result with the optimal coefficients obtained by *least squares*. The code below shows how *training loss* is  computed.

The most specific part of the algorithm is the gradient computation. Note that the *gradient machinery* of `PyTorch` is turned-on for each weight with `requires_grad = True` as in the following case:

    coeffs=torch.tensor([-20.,-10.]).requires_grad_()

Then, the derivatives can be computed for any continuous function of the weights in tensor `coeffs`. In particular, the *loss* $L$ is defined as a function (that can be arbitrarily complicated) of the weights, and the *gradient* $\nabla L({\rm \bf w}^{*})$ for the current set of weights ${\rm \bf w}^{*}$ is computed with


    loss.backward()


Finally, the weights are updated with

    coeffs.sub_(coeffs.grad * step_size)

where method `sub_` is substraction for weight updating ${\rm \bf w}^{*}:={\rm \bf w}^{*} - \eta \, \nabla L({\rm \bf w}^{*})$, and  the learning rate $\eta$ is called `step_size` in the code.

Try changing the learning rate to see what happens (try for instance `step_size=0.1`).


```python
#@title Script for stochastic gradient descent with Pytorch, train only data, applied to synthetic LR data
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
    return torch.mean((y_pred - y) ** 2) # mean applies to a single value in this case

# lists to store losses for each epoch
training_losses = []

# epochs
for i in range(iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = calc_preds(X)
    loss = calc_loss_from_labels(y_pred, y)
    training_losses.append(loss.item())

    # Stochastic Gradient Descent (SGD): update weights after each data point
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
            coeffs.grad.zero_() # PyTorch accumulates the gradients on subsequent backward passes. So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.

print('coeffs found by stochastic gradient descent:', coeffs.detach().numpy())

# plot training loss along epochs
plt.plot(training_losses, '-g')
plt.xlabel('epoch')
plt.ylabel('loss (MSE)')
plt.show()

```
