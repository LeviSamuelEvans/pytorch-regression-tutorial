# Theory
This document contains a brief overview of some of the 'theory' behind the applied ML tutorial.

- [Theory](#theory)
  - [Gradient Descent](#gradient-descent)
    - [Overview](#overview)
      - [Step 1: Define and compute the Loss Function](#step-1-define-and-compute-the-loss-function)
      - [Step 2: Compute the Gradient](#step-2-compute-the-gradient)
      - [Step 3: Update the Parameters](#step-3-update-the-parameters)
    - [Recap](#recap)
  - [Weight Initialisation](#weight-initialisation)

## Gradient Descent

Gradient descent is the mechanism behind the training of many machine learning models. It is a type of optimisation algorithm that finds the mininum of some function by iteratively moving towards the negative gradient of the function. It works in the following manner:

1. Initialise the parameters (weights) of the model.
2. Compute the loss function.
3. Compute the gradient (partial derivatives) of the loss function with respect to each parameter.
4. Update the parameters using the gradients.
5. Repeat until convergence.

### Overview

Let us start from the loss function we use in this tutorial:

#### Step 1: Define and compute the Loss Function

For linear regression, the MSE loss function (`mean squared error`) is:
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 $$

where:
- $\theta$ are the parameters (weights).
- $h_{\theta}(x)$ is the hypothesis (prediction) $h_{\theta}(x) = \theta_0 + \theta_1 x$.
- $m$ is the number of training examples.
- $x^{(i)}$ and $y^{(i)}$ are the input and output of the $i$-th training example.
  
#### Step 2: Compute the Gradient

The gradient of the loss function with respect to $\theta_0$ and $\theta_1$ is:
$$ \frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) $$
$$ \frac{\partial J(\theta)}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x^{(i)} $$


#### Step 3: Update the Parameters

The parameters are updated as follows:
$$ \theta_0 := \theta_0 - \alpha \frac{\partial J(\theta)}{\partial \theta_0} $$
$$ \theta_1 := \theta_1 - \alpha \frac{\partial J(\theta)}{\partial \theta_1} $$

where $\alpha$ is the `learning rate`, which is a hyperparameter that controls the `step size` of the gradient descent.

### Recap

The gradient descent method can be summarised as follows:

1. **Initialisation**: We start with an initial guess for the parameters $\theta_0$ and $\theta_1$.

2. **Predict**: Compute the predictions for all training examples using the current values of $\theta_0$ and $\theta_1$.

3. **Compute Cost**: Calculate the loss function $J(\theta)$ to see how well the model our performing.

4. **Calculate Gradient**: Determine the gradients $\frac{\partial J(\theta)}{\partial \theta_0}$ and $\frac{\partial J(\theta)}{\partial \theta_1}$. These tell us how to adjust $\theta_0$ and $\theta_1$ to reduce the value of the loss function.

5. **Update Parameters**: Adjust the parameters $\theta_0$ and $\theta_1$ by moving them a small step in the direction opposite to the gradient, where the size of the step is controlled by the learning rate $\alpha$.

6. **Repeat**: Repeat the process until the loss function converges to a minimum value, indicating that we have found the optimal parameters.

## Weight Initialisation