# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

# # Neural Network Fundamentals
# This notebook introduces fundamental concepts for building neural networks step by step

# # Part 1: Understanding the Perceptron
# We'll examine how a neuron works by building it from basic components

# ## Level 1: Basic Weighted Sum
# First, let's define our input values and weights

# %%
# Input values
x1, x2, x3 = 0, 2, 0

# %%
# Weights
w1, w2, w3 = -1, 1, 3

# %%
# Bias/threshold value
b = 3

# %%
# Calculate the weighted sum of inputs
output = x1*w1 + x2*w2 + x3*w3
output

# %%
# Apply threshold function (activation)
if output >= b:
    result = 1
else:
    result = 0 

# %%
result

# ## Level 2: Treating Bias as a Weight
# We can incorporate the bias into our weights by using a constant input

# %%
# Same inputs and weights as before
x1, x2, x3 = 0, 2, 0
w1, w2, w3 = -1, 1, 3
b = 3

# %%
# Check if the weighted sum minus bias is positive
x1*w1 + x2*w2 + x3*w3 - 1*b >= 0

# %%
# Turn bias into a weight
w4 = b

# %%
# Calculate result with bias as a weight
if x1*w1 + x2*w2 + x3*w3 - 1*w4 >= 0:
    result = 1
else:
    result = 0

# ## Level 3: Using Lists for Inputs and Weights
# Let's use lists to handle multiple inputs/weights more elegantly

# %%
# Combine inputs and weights into lists
x = [x1, x2, x3, 1]  # Note: Added constant input 1
w = [w1, w2, w3, w4]

# %%
# Sum products with basic loop
s = 0
for i in range(len(x)):
    s += x[i]*w[i]  # Fixed index bug from original

# %%
(s >= 0)*1  # Multiply by 1 to convert boolean to int

# ## Level 4: Using Zip for Cleaner Code
# Python's zip function makes this process more elegant

# %%
# Reset sum and use zip to pair inputs with weights
s = 0
for x_n, w_n in zip(x, w):
    s += x_n*w_n

# %%
(s >= 0)*1

# ## Level 5: Using List Comprehension
# List comprehension provides an even more concise approach

# %%
# Calculate each product with list comprehension
tmp_s = [x_n*w_n for x_n, w_n in zip(x, w)]

tmp_s

# %%
# Sum the products
s = sum(tmp_s)

# %%
(s >= 0)*1

# ## Level 6: Using NumPy for Vector Operations
# NumPy provides efficient vector operations

# %%
import numpy as np

# %%
# Use NumPy's dot product function
(np.dot(x, w) >= 0)*1

# %%
# Convert to NumPy arrays
xa = np.array(x)
wa = np.array(w)

# %%
# Use matrix multiplication operator
xa@wa

# ## Level 7: Creating a Neuron Function
# Let's wrap our neuron logic in a reusable function

# %%
def neuron(x, w):
    """
    Implements a simple neuron with step activation function
    x: input values (with bias term)
    w: weights (including bias weight)
    """
    return ((x@w) >= 0)*1

# # Part 2: Logical Operations with Neurons
# We'll implement logical operations using our neuron model

# ## Level 1: Basic Neuron Test
# Testing the neuron with basic inputs

# %%
# Create simple test inputs
x = np.array([0, 0, 1])  # Added bias term
w = np.array([0, 0, 0])

# %%
neuron(x, w)

# ## Level 2: OR Gate Implementation
# Implementing the OR logical operation

# %%
# Truth table for OR operation
X = np.array([
    [0, 0, 1],  # [in1, in2, bias]
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],  # Expected outputs
    [1],
    [1],
    [1]
])

# %%
# Weights for OR operation
w = np.array([1, 1, -0.5])

# %%
# Test OR implementation
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# ## Level 3: AND Gate Implementation
# Implementing the AND logical operation

# %%
# Using same truth table setup, just changing the weights
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# %%
# Weights for AND operation
w = np.array([1, 1, -1.5])

# %%
# Test AND implementation
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# ## Level 4: XOR Problem
# Demonstrating the XOR problem (not linearly separable)

# %%
# Truth table for XOR
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# %%
# Attempt with simple weights (will not work completely)
w = np.array([-0.5, 1, -1])

# %%
# Test XOR implementation (will show the failure of a single neuron)
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# # Part 3: Linear Regression with a Neuron
# Using a neuron without the step function for linear regression

# ## Level 1: Creating a Linear Neuron
# The key difference from our previous neuron is removing the threshold activation

# %%
import matplotlib.pyplot as plt

# %%
# Define a linear neuron (without activation)
def neuron2(x, w):
    """Linear neuron - just returns the weighted sum"""
    return x@w

# ## Level 2: Generating Linear Data
# Creating a simple dataset to visualize linear models

# %%
# Generate x values for our data
x_1 = np.linspace(0, 10, 25)  # 25 points between 0 and 10

# %%
# Add bias term (constant 1)
x_2 = np.zeros_like(x_1) + 1  # Bias term

# %%
# Stack into input matrix with shape (25, 2)
X = np.vstack((x_1, x_2)).T

# %%
# Examine first few rows of our data
X[:5]

# ## Level 3: Creating Linear Models
# Different weights create different linear functions

# %%
# First set of weights: steeper slope
w1 = np.array([0.5, 1])  # slope 0.5, intercept 1
Y1 = neuron2(X, w1)

# %%
# Second set of weights: gentler slope
w2 = np.array([0.3, 1])  # slope 0.3, intercept 1
Y2 = neuron2(X, w2)

# %%
# Plot both lines to compare
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], Y1, label="Line 1: slope=0.5, intercept=1")
plt.plot(X[:, 0], Y2, label="Line 2: slope=0.3, intercept=1")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Functions with Different Slopes")
plt.grid(True)