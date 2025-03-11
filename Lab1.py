{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "68e3c524",
   "metadata": {},
   "source": [
    "# Neural Network Fundamentals\n",
    "This notebook introduces fundamental concepts for building neural networks step by step"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "144c0428",
   "metadata": {},
   "source": [
    "# Part 1: Understanding the Perceptron\n",
    "We'll examine how a neuron works by building it from basic components"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a5b6f32",
   "metadata": {},
   "source": [
    "## Level 1: Basic Weighted Sum\n",
    "First, let's define our input values and weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "650f1ecd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Input values\n",
    "x1, x2, x3 = 0, 2, 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e55922d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Weights\n",
    "w1, w2, w3 = -1, 1, 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2ee541b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bias/threshold value\n",
    "b = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7a5f9f68",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# %%\n",
    "# Calculate the weighted sum of inputs\n",
    "output = x1*w1 + x2*w2 + x3*w3\n",
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "da16d7e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Apply threshold function (activation)\n",
    "if output >= b:\n",
    "    result = 1\n",
    "else:\n",
    "    result = 0 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b9d2e01b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a7a3a06",
   "metadata": {},
   "source": [
    "## Level 2: Treating Bias as a Weight\n",
    "We can incorporate the bias into our weights by using a constant input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ffe71a5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Same inputs and weights as before\n",
    "x1, x2, x3 = 0, 2, 0\n",
    "w1, w2, w3 = -1, 1, 3\n",
    "b = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6fe26afc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# %%\n",
    "# Check if the weighted sum minus bias is positive\n",
    "x1*w1 + x2*w2 + x3*w3 - 1*b >= 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8b81d3ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Turn bias into a weight\n",
    "w4 = b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1bbea3d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Calculate result with bias as a weight\n",
    "if x1*w1 + x2*w2 + x3*w3 - 1*w4 >= 0:\n",
    "    result = 1\n",
    "else:\n",
    "    result = 0"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8509aa5c",
   "metadata": {},
   "source": [
    "## Level 3: Using Lists for Inputs and Weights\n",
    "Let's use lists to handle multiple inputs/weights more elegantly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "3fa5979e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Combine inputs and weights into lists\n",
    "x = [x1, x2, x3, 1]  # Note: Added constant input 1\n",
    "w = [w1, w2, w3, w4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "83440d7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Sum products with basic loop\n",
    "s = 0\n",
    "for i in range(len(x)):\n",
    "    s += x[i]*w[i]  # Fixed index bug from original"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7e9a2e5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(s >= 0)*1  # Multiply by 1 to convert boolean to int"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f773a31",
   "metadata": {},
   "source": [
    "## Level 4: Using Zip for Cleaner Code\n",
    "Python's zip function makes this process more elegant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c40e1094",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Reset sum and use zip to pair inputs with weights\n",
    "s = 0\n",
    "for x_n, w_n in zip(x, w):\n",
    "    s += x_n*w_n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6ca6fbbd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(s >= 0)*1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "214a5f22",
   "metadata": {},
   "source": [
    "## Level 5: Using List Comprehension\n",
    "List comprehension provides an even more concise approach"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "587f5500",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0, 2, 0, 3]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculate each product with list comprehension\n",
    "tmp_s = [x_n*w_n for x_n, w_n in zip(x, w)]\n",
    "\n",
    "tmp_s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "99b98f82",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Sum the products\n",
    "s = sum(tmp_s)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "529e7d0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(s >= 0)*1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6695d237",
   "metadata": {},
   "source": [
    "## Level 6: Using NumPy for Vector Operations\n",
    "NumPy provides efficient vector operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "73354144",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a8b13ed3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Use NumPy's dot product function\n",
    "(np.dot(x, w) >= 0)*1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ce9b5d8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Convert to NumPy arrays\n",
    "xa = np.array(x)\n",
    "wa = np.array(w)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "7320bbeb",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Use matrix multiplication operator\n",
    "xa@wa"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1025871",
   "metadata": {},
   "source": [
    "## Level 7: Creating a Neuron Function\n",
    "Let's wrap our neuron logic in a reusable function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e73938cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "def neuron(x, w):\n",
    "    \"\"\"\n",
    "    Implements a simple neuron with step activation function\n",
    "    x: input values (with bias term)\n",
    "    w: weights (including bias weight)\n",
    "    \"\"\"\n",
    "    return ((x@w) >= 0)*1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07dc38c3",
   "metadata": {},
   "source": [
    "# Part 2: Logical Operations with Neurons\n",
    "We'll implement logical operations using our neuron model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e815401d",
   "metadata": {},
   "source": [
    "## Level 1: Basic Neuron Test\n",
    "Testing the neuron with basic inputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8680ac76",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Create simple test inputs\n",
    "x = np.array([0, 0, 1])  # Added bias term\n",
    "w = np.array([0, 0, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "54c8d250",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "neuron(x, w)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47dffdef",
   "metadata": {},
   "source": [
    "## Level 2: OR Gate Implementation\n",
    "Implementing the OR logical operation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "5574629e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Truth table for OR operation\n",
    "X = np.array([\n",
    "    [0, 0, 1],  # [in1, in2, bias]\n",
    "    [1, 0, 1],\n",
    "    [0, 1, 1],\n",
    "    [1, 1, 1]\n",
    "])\n",
    "\n",
    "Y = np.array([\n",
    "    [0],  # Expected outputs\n",
    "    [1],\n",
    "    [1],\n",
    "    [1]\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "09d28c0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Weights for OR operation\n",
    "w = np.array([1, 1, -0.5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7e69da34",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: [0 0], Output: 0, Expected: 0\n",
      "Input: [1 0], Output: 1, Expected: 1\n",
      "Input: [0 1], Output: 1, Expected: 1\n",
      "Input: [1 1], Output: 1, Expected: 1\n"
     ]
    }
   ],
   "source": [
    "# %%\n",
    "# Test OR implementation\n",
    "for i in range(4):\n",
    "    print(f\"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "daa2cc55",
   "metadata": {},
   "source": [
    "## Level 3: AND Gate Implementation\n",
    "Implementing the AND logical operation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "362b70bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using same truth table setup, just changing the weights\n",
    "X = np.array([\n",
    "    [0, 0, 1],\n",
    "    [1, 0, 1],\n",
    "    [0, 1, 1],\n",
    "    [1, 1, 1]\n",
    "])\n",
    "\n",
    "Y = np.array([\n",
    "    [0],\n",
    "    [0],\n",
    "    [0],\n",
    "    [1]\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5ca928bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Weights for AND operation\n",
    "w = np.array([1, 1, -1.5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f808f580",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: [0 0], Output: 0, Expected: 0\n",
      "Input: [1 0], Output: 0, Expected: 0\n",
      "Input: [0 1], Output: 0, Expected: 0\n",
      "Input: [1 1], Output: 1, Expected: 1\n"
     ]
    }
   ],
   "source": [
    "# %%\n",
    "# Test AND implementation\n",
    "for i in range(4):\n",
    "    print(f\"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "199e64cc",
   "metadata": {},
   "source": [
    "## Level 4: XOR Problem\n",
    "Demonstrating the XOR problem (not linearly separable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9dbe2d26",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Truth table for XOR\n",
    "X = np.array([\n",
    "    [0, 0, 1],\n",
    "    [1, 0, 1],\n",
    "    [0, 1, 1],\n",
    "    [1, 1, 1]\n",
    "])\n",
    "\n",
    "Y = np.array([\n",
    "    [0],\n",
    "    [1],\n",
    "    [1],\n",
    "    [0]\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "19ef52c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Attempt with simple weights (will not work completely)\n",
    "w = np.array([-0.5, 1, -1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "eb3b1161",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input: [0 0], Output: 0, Expected: 0\n",
      "Input: [1 0], Output: 0, Expected: 1\n",
      "Input: [0 1], Output: 1, Expected: 1\n",
      "Input: [1 1], Output: 0, Expected: 0\n"
     ]
    }
   ],
   "source": [
    "# %%\n",
    "# Test XOR implementation (will show the failure of a single neuron)\n",
    "for i in range(4):\n",
    "    print(f\"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4234882a",
   "metadata": {},
   "source": [
    "# Part 3: Linear Regression with a Neuron\n",
    "Using a neuron without the step function for linear regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb2bec7e",
   "metadata": {},
   "source": [
    "## Level 1: Creating a Linear Neuron\n",
    "The key difference from our previous neuron is removing the threshold activation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "3a14b938",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "# %%\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a3875644",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "# %%\n",
    "# Define a linear neuron (without activation)\n",
    "def neuron2(x, w):\n",
    "    \"\"\"Linear neuron - just returns the weighted sum\"\"\"\n",
    "    return x@w"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e42679e2",
   "metadata": {},
   "source": [
    "## Level 2: Generating Linear Data\n",
    "Creating a simple dataset to visualize linear models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0025458f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Generate x values for our data\n",
    "x_1 = np.linspace(0, 10, 25)  # 25 points between 0 and 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "6911359e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Add bias term (constant 1)\n",
    "x_2 = np.zeros_like(x_1) + 1  # Bias term"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "aa3e7786",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Stack into input matrix with shape (25, 2)\n",
    "X = np.vstack((x_1, x_2)).T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "d5c29e82",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.        , 1.        ],\n",
       "       [0.41666667, 1.        ],\n",
       "       [0.83333333, 1.        ],\n",
       "       [1.25      , 1.        ],\n",
       "       [1.66666667, 1.        ]])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# %%\n",
    "# Examine first few rows of our data\n",
    "X[:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9dd221bf",
   "metadata": {},
   "source": [
    "## Level 3: Creating Linear Models\n",
    "Different weights create different linear functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "b494bf96",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# First set of weights: steeper slope\n",
    "w1 = np.array([0.5, 1])  # slope 0.5, intercept 1\n",
    "Y1 = neuron2(X, w1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "4305bea4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %%\n",
    "# Second set of weights: gentler slope\n",
    "w2 = np.array([0.3, 1])  # slope 0.3, intercept 1\n",
    "Y2 = neuron2(X, w2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "fe1a333b",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0EAAAIjCAYAAADFthA8AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAACZ8ElEQVR4nOzdd3hUZfrG8e9Meg8khFBCQgmEJHQLvUjvooJdcW1rr8gqCgQL6k93QV3RXRWsq6IgRXoHKzYgCQQCofeSTpIp5/fHQIZIMQHCSbk/18Ulec+ZmWdOXuLcOe95jsUwDAMREREREZFqwmp2ASIiIiIiIpeSQpCIiIiIiFQrCkEiIiIiIlKtKASJiIiIiEi1ohAkIiIiIiLVikKQiIiIiIhUKwpBIiIiIiJSrSgEiYiIiIhItaIQJCIiIiIi1YpCkIiYZvv27VgsFqZNm2Z2KdXGihUrsFgsrFixwuxSLppp06ZhsVjYvn17qff95Zdfyr+wE0aOHElMTEyJsdzcXO666y4iIyOxWCw8+uijABw4cIDrrruOsLAwLBYLkyZNumR1VhTjx4/HYrGYXYaIVHEKQSJSLsz4sHmpnfywdqY/77zzjqm1vf3229U6XJbX+//z99zf358GDRowePBgpk6dSmFhYame56WXXmLatGncd999fPzxx9x6660APPbYYyxcuJCnn36ajz/+mH79+l3093CxlPUY5+bmMm7cOBITEwkICCAsLIzWrVvzyCOPsHfv3vIrVETkDDzNLkBEqq/o6GiOHz+Ol5eX2aVckClTphAYGFhi7MorrzSpGpe3336b8PBwRo4cWWK8a9euHD9+HG9vb3MKKwe33norN9xwAz4+PsVjZ3v/F8vJ73lhYSF79uxh4cKF/O1vf2PSpEnMnTuXqKio4n3/+9//4nQ6Szx+2bJltG/fnnHjxp02PnToUJ588slyqftiKssxttlsdO3alU2bNnH77bfz0EMPkZubS0pKCp999hnDhg2jbt265V+0iMgJCkEiYhqLxYKvr6/ZZZxTfn4+/v7+59znuuuuIzw8/BJVdGGsVmuFP+Zl5eHhgYeHxyV9zT9/z8eOHcunn37KbbfdxvDhw/nxxx+Lt50p5B88eJD4+PgzjoeGhl60Ou12O06n0/TQ+8033/D777/z6aefctNNN5XYVlBQQFFRkUmViUh1peVwImKaM10TNHLkSAIDA9mzZw9XX301gYGB1KpViyeffBKHw1Hi8U6nk0mTJpGQkICvry+1a9fm3nvv5dixYyX2mzVrFgMHDqRu3br4+PjQuHFjnn/++dOer3v37iQmJvLrr7/StWtX/P39eeaZZy7q+zvJYrEwfvz44q9PLrNKT09n5MiRhIaGEhISwh133EF+fv5pj//kk0+44oor8Pf3p0aNGnTt2pVFixYBEBMTQ0pKCitXrixettW9e3fg7NcETZ8+nXbt2uHn50d4eDi33HILe/bsKbFPWb43n3/+Oe3atSMoKIjg4GBatGjB5MmTz3m82rZtyzXXXFNirEWLFlgsFtavX1889sUXX2CxWNi4cSNw+jVB53r/JxUWFvL4449Tq1YtAgICGDZsGIcOHTpnfX/l5ptv5q677uKnn35i8eLFxeOnXhN08vhnZGTw7bffFtd38j0YhsG///3v4vGTMjMzefTRR4mKisLHx4cmTZrwyiuvlDjDdHK+vfbaa0yaNInGjRvj4+NDamoqAJs2beK6666jZs2a+Pr6ctlllzF79uwS7+FkHd999905j09pjvGptm7dCkCnTp1O2+br60twcPA5j63dbuf5558vfk8xMTE888wzpy0/jImJYdCgQSxatIjWrVvj6+tLfHw8M2bMOO05S3NM4fzmsohUfApBIlLhOBwO+vbtS1hYGK+99hrdunXj9ddf5z//+U+J/e69915GjRpFp06dmDx5MnfccQeffvopffv2xWazFe83bdo0AgMDefzxx5k8eTLt2rVj7Nix/OMf/zjttY8cOUL//v1p3bo1kyZNokePHn9Z79GjRzl8+HDxnz+HsLIYMWIEOTk5TJw4kREjRjBt2jSSkpJK7JOUlMStt96Kl5cXEyZMICkpiaioKJYtWwbApEmTqF+/PnFxcXz88cd8/PHHjBkz5qyvOW3aNEaMGIGHhwcTJ07k7rvvZsaMGXTu3JnMzMwS+5bme7N48WJuvPFGatSowSuvvMLLL79M9+7d+e6778753rt06cKaNWuKvz569CgpKSlYrVZWr15dPL569Wpq1apF8+bNz/g8pXn/Dz30EOvWrWPcuHHcd999zJkzhwcffPCc9ZXGyWt7TgbSP2vevDkff/wx4eHhtG7duri+yy+/nI8//hiA3r17F4+D62xkt27d+OSTT7jtttt4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dDROp5MtW7aUaIZx4MABMjMzT5ujJ898nVrf5s2bAYq79JXlmJ7vXBaRik3XBIlIpTRixAgcDgfPP//8advsdntx8Dj5YfnU32AXFRXx9ttvl3uNwcHBhIeHn3bNxoW89tVXX43VamXChAmnncU69T0GBAScFr7O5LLLLiMiIoJ33nmnxHUZ8+fPZ+PGjQwcOLDMNR45cqTE11arlZYtWwKc89oPcF8X9Morr9CyZUtCQkKKx5cuXcovv/xSqqVwpX3/F9Nnn33Ge++9R4cOHejZs+dFe94RI0bwww8/sHDhwtO2ZWZmYrfbz/n4iIgIunfvzrvvvsu+fftO236+rcHLcozXrVvH4cOHTxvfsWMHqampZ1yOedKAAQMAV0e6U/3zn/8EOG2O7t27t0THu+zsbD766CNat25NZGQkUPpjeiFzWUQqNp0JEpFy9cEHH7BgwYLTxh955JELet5u3bpx7733MnHiRP744w/69OmDl5cXW7ZsYfr06UyePJnrrruOjh07UqNGDW6//XYefvhhLBYLH3/88SVbXnbXXXfx8ssvc9ddd3HZZZexatWq4t9Kn48mTZowZswYnn/+ebp06cI111yDj48Pa9eupW7dukycOBGAdu3aMWXKFF544QWaNGlCRETEaWd6wHV24JVXXuGOO+6gW7du3HjjjRw4cIDJkycTExPDY489dl7v+ejRo1x11VXUr1+fHTt28Oabb9K6deuztrU+9f1FRkaSlpbGQw89VDzetWtXRo8eDVCqEFTa93++vvrqKwIDAykqKmLPnj0sXLiQ7777jlatWjF9+vSL9joAo0aNYvbs2QwaNIiRI0fSrl078vLy2LBhA1999RXbt2//y2vS/v3vf9O5c2datGjB3XffTaNGjThw4AA//PADu3fvZt26dWWuqyzHePHixYwbN44hQ4bQvn17AgMD2bZtGx988AGFhYUl7pn1Z61ateL222/nP//5T/Hy1p9//pkPP/yQq6+++rQ29k2bNuXOO+9k7dq11K5dmw8++IADBw4wderU4n1Ke0wvZC6LSMWmECQi5WrKlClnHB85cuQFP/c777xDu3btePfdd3nmmWfw9PQkJiaGW265pfimjGFhYcydO5cnnniCZ599lho1anDLLbfQs2fPi7pk6WzGjh3LoUOH+Oqrr4ovsJ4/f/5ZGzqUxoQJE2jYsCFvvvkmY8aMwd/fn5YtWxbfo+bk6+7YsYNXX32VnJwcunXrdtYPqCNHjsTf35+XX36Z0aNHF98c85VXXiE0NLTM9d1yyy385z//4e233yYzM5PIyEiuv/56xo8ff1pHuzPp0qUL06dPp3PnzsVj7dq1w9/fH7vdzpVXXvmXz1GW938+7rvvPsDVKOLkPX8++OADbrrpJnx8fC7a6wD4+/uzcuVKXnrpJaZPn85HH31EcHAwTZs2JSkpqfhs2bnEx8fzyy+/kJSUxLRp0zhy5AgRERG0adOGsWPHnlddZTnG1157LTk5OSxatIhly5Zx9OhRatSowRVXXMETTzzxl/fjeu+992jUqBHTpk1j5syZREZG8vTTTzNu3LjT9o2NjeXNN99k1KhRpKWl0bBhQ7744osS/95Le0wvdC6LSMVlMSrK1bYiIiIiFyAmJobExETmzp1rdikiUsHp1xgiIiIiIlKtKASJiIiIiEi1ohAkIiIiIiLViq4JEhERERGRakVngkREREREpFpRCBIRERERkWqlUt8nyOl0snfvXoKCgrBYLGaXIyIiIiIiJjEMg5ycHOrWrfuX9/Kq1CFo7969REVFmV2GiIiIiIhUELt27aJ+/frn3KdSh6CgoCDA9UaDg4NNrcVms7Fo0SL69OmDl5eXqbVI5aA5I2WlOSNlpTkjZaU5I2VVkeZMdnY2UVFRxRnhXCp1CDq5BC44OLhChCB/f3+Cg4NNnwBSOWjOSFlpzkhZac5IWWnOSFlVxDlTmstk1BhBRERERESqFYUgERERERGpVhSCRERERESkWqnU1wSVhmEY2O12HA5Hub6OzWbD09OTgoKCcn8tqRo0Z8zh5eWFh4eH2WWIiIiIiap0CCoqKmLfvn3k5+eX+2sZhkFkZCS7du3SPYukVDRnzGGxWKhfvz6BgYFmlyIiIiImqbIhyOl0kpGRgYeHB3Xr1sXb27tcP2g6nU5yc3MJDAz8y5sziYDmjBkMw+DQoUPs3r2b2NhYnRESERGppqpsCCoqKsLpdBIVFYW/v3+5v57T6aSoqAhfX199oJVS0ZwxR61atdi+fTs2m00hSEREpJqq8p+89OFSRE6lpYciIiKihCAiIiIiItWKQpCIiIiIiFQrCkGVkMVi4ZtvvjG7jAs2fvx4WrdubXYZFZaOj4iIiEj5UAiqgEaOHMnVV1991u379u2jf//+5VrDvn37uOmmm2jatClWq5VHH320XF+vIjt69Cg333wzwcHBhIaGcuedd5Kbm3vOx3Tv3h2LxVLiz9///vcyve6TTz7J0qVLy/SYmJgYJk2aVKbHmGnatGmEhoZe8PM8/PDDtGvXDh8fHwVHERER+UsKQZVQZGQkPj4+5foahYWF1KpVi2effZZWrVqV62tVdDfffDMpKSksXryYuXPnsmrVKu65556/fNzdd9/Nvn37iv+8+uqrZXrdwMBAwsLCzrfsC1JUVGTK616Iv/3tb1x//fVmlyEiIiKVQLUKQYZhkF9kL7c/x4scZxw3DOOivo9Tl8Nt374di8XCjBkz6NGjB/7+/rRq1YoffvihxGPWrFlDly5d8PPzIyoqiocffpi8vLyzvkZMTAyTJ0/mtttuIyQk5LxrXbFiBVdccQUBAQGEhobSqVMnduzYccZ9nU4nEyZMoH79+sW/0V+wYEHx9pPv9fPPP6djx474+vqSmJjIypUrSzxPcnIy/fv3JzAwkNq1a3Prrbdy+PDh86p/48aNLFiwgPfee48rr7ySzp078+abb/L555+zd+/ecz7W39+fyMjI4j/BwcFleu0/L4c7eYbwtddeo06dOoSFhfHAAw9gs9kA19mnHTt28NhjjxWffTrpr77/MTExPP/889x2220EBwcXh7zvvvuO7t274+/vT40aNejbty/Hjh0DXN+viRMn0rBhQ/z8/GjVqhVfffVV8XOuWLECi8XCt99+S8uWLfH19aV9+/YkJycXb7/jjjvIysoqrnf8+PFlOkYnvfHGGzzwwAM0atTovB4vIiIi1Yvp9wnas2cPo0ePZv78+eTn59OkSROmTp3KZZdddtFf67jNQfzYhRf9ef9K6oS++HuX76EeM2YMr732GrGxsYwZM4Ybb7yR9PR0PD092bp1K/369eOFF17ggw8+4NChQzz44IM8+OCDTJ069bxfc8WKFfTo0YOMjAxiYmJO226327n66qu5++67+d///kdRURE///zzWVsUT548mddff513332XNm3a8MEHHzBkyBBSUlKIjY0t3m/UqFFMmjSJ+Ph4/vnPfzJ48GAyMjIICwsjMzOTq666irvuuot//etfHD9+nNGjRzNixAiWLVsGwEsvvcRLL710zveWmppKgwYN+OGHHwgNDS0xH3v16oXVauWnn35i2LBhZ32OTz/9lE8++YTIyEgGDx7Mc889d8H3rFq+fDl16tRh+fLlpKenc/3119O6dWvuvvtuZsyYQatWrbjnnnu4++67ix9T2u//a6+9xtixYxk3bhwAf/zxBz179uRvf/sbkydPxtPTk+XLl+NwOACYOHEin3zyCe+88w6xsbGsWrWKW265hVq1atGtW7fi5x01ahSTJ08mMjKSZ555hsGDB7N582Y6duzIpEmTGDt2LGlpaYDr7BfA3//+dz755JNzHou/WpIoIiIicjamhqBjx47RqVMnevTowfz586lVqxZbtmyhRo0aZpZVKT355JMMHDgQgKSkJBISEkhPTycuLo6JEydy8803F1/XExsbyxtvvEG3bt2YMmUKvr6+5/Wa/v7+NGvWDC8vrzNuz87OJisri0GDBtG4cWMAmjdvftbne+211xg9ejQ33HADAK+88grLly9n0qRJ/Pvf/y7e78EHH+Taa68FYMqUKSxYsID333+fp556irfeeos2bdqUCDkffPABUVFRbN68maZNm/L3v/+dESNGnPO91a1bF4D9+/cTERFRYpunpyc1a9Zk//79Z338TTfdRHR0NHXr1mX9+vWMHj2atLQ0ZsyYcc7X/Ss1atTgrbfewsPDg7i4OAYOHMjSpUu5++67qVmzJh4eHgQFBREZGVn8mNJ+/6+66iqeeOKJEu/hsssu4+233y4eS0hIAFzLJV966SWWLFlChw4dAGjUqBFr1qzh3XffLRGCxo0bR+/evQH48MMPqV+/PjNnzmTEiBGEhIRgsVhK1AswYcIEnnzyyQs6ViIiIiJnY2oIeuWVV4iKiirx2+iGDRuW2+v5eXmQOqFvuTy30+kkJzuHoOCg027Q6udV/nelb9myZfHf69SpA8DBgweJi4tj3bp1rF+/nk8//bR4H8MwcDqdZGRknDOYnMsVV1zBpk2bzrq9Zs2ajBw5kr59+9K7d2969erFiBEjius7VXZ2Nnv37qVTp04lxjt16sS6detKjJ380A2uQHLZZZexceNGANatW8fy5cuLzyicauvWrTRt2pSaNWtSs2bNMr3Xsjr1mqEWLVpQp04devbsydatW4sD4flISEjAw8M9n+rUqcOGDRvO+ZjSfv//fPb1jz/+YPjw4Wd8zvT0dPLz84vDzUlFRUW0adOmxNip36+aNWvSrFmz4u/X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lkf9WatWrejZsyctWrSgb9++9OnTh+uuu44aNWqctu/WrVux2Wx06tSpeMzLy4srrriCjRs3lti3Q4cOxX+vWbMmzZo1K96nNO81IiKixPsoL6NGjeLOO+9kx44dJCUlcdtttzF37twLWo976vcZXN/rgwcPnvMx69at47vvvite4gau62gKCgrIz8/H3981Py+77LISj/vjjz+4++67z/icGzZswOFw0LRp0xLjhYWFhIWFFX/t6enJ5ZdfXvx1XFwcoaGhbNy4kSuuuOKsNderV++c70lERKS6cjoNvv5tN68s2MThXNfS/AEtIhkzMJ56oX4mV1d5VK8QZLGU37I0pxO8HK7n/3MIugRO/c34yQ/ZTqcTcC3tuvfee3n44YdPe1yDBg3O+byff/45d911F9OnTy/zReceHh4sXryY77//nkWLFvHmm28yZswYfvrpJxo2bFim5yqt0rzXsiyHi4yM5Oeffy6x7cCBA8XbziU8PJzw8HCaNm1K8+bNiYqK4scffywR4srqz2dALBZL8ff5bHJzc0lKSuKaa645bdupZwEDAkr+2/DzO/sP0tzcXDw8PPj111+Ll92dFBh44Xec1nI4ERGR0yXvyWLsrGR+25kJQONaASQNSaRzbLi5hVVC1SsEVVNt27YlNTWVJk2alOlx//vf//jb3/7G559/XrzUrqwsFgudOnWiU6dOjB07lujoaGbOnMnjjz9eYr/GjRvj7e3Nd999V7x0zmazsXbt2uKlbSf9+OOPxYHm2LFjbN68ufhsVmnea1mWw3Xo0IEXX3yRgwcPFp89Wrx4McHBwcTHx5fqOcAdSAsLC0v9mPPh5eWFw+EoMda2bVvS0tLK/P1v2bIlS5cuLbFE7qQ2bdrgcDg4ePAgXbp0Oetz2O12fvnll+KzPmlpaWRmZhZ/v7y9vU+rF7QcTkRE5FSZ+UX838I0Pvt5J4YBAd4ePNIrlpEdG+LtWb36nF0sCkEVVFZWFn/88UeJsbCwMKKiosr8XKNHj6Z9+/Y8+OCD3HXXXQQEBJCamsrixYt56623zviYzz77jNtvv53Jkydz5ZVXFl//4ufnV3zx+syZM3n66afZtGnTGZ/jp59+YunSpfTp04eIiAh++uknDh06dMbldwEBAdx3332MGjWKmjVr0qBBA1599VXy8/O58847S+w7YcIEwsLCqF27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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# %%\n",
    "# Plot both lines to compare\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(X[:, 0], Y1, label=\"Line 1: slope=0.5, intercept=1\")\n",
    "plt.plot(X[:, 0], Y2, label=\"Line 2: slope=0.3, intercept=1\")\n",
    "plt.legend()\n",
    "plt.xlabel(\"x\")\n",
    "plt.ylabel(\"y\")\n",
    "plt.title(\"Linear Functions with Different Slopes\")\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2869471a-2598-4b06-8c9b-e4a11ccfa07e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7aa98169-9429-4515-97b1-62167f8f7fab",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "jupytext": {
   "formats": "ipynb,py:light"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
