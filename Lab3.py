# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd

boston = fetch_openml(name="boston", version=1, as_frame=True)

df = boston.frame

plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
#x_1 = np.linspace(df['LSTAT'].min(), df["LSTAT"].max())
x_1 = df["LSTAT"]
x_b = np.zeros_like(x_1)
x = np.vstack([x_1,x_b]).T
w = np.array([0.3,1])
y_pred = neuron2(x, w)
error = (df["MEDV"].values - y_pred)
mse = np.mean(error**2)
plt.title(mse)
plt.plot(x_1,y_pred, c='r')
plt.scatter(df['LSTAT'][0], df['MEDV'][0], color="black")



def correct_weights(x_n, y_n, w):
    w_0 = (y_n + w[1])/x_n
    w_1 = w_0*x_n-y_n
    return w_0, w_1


w_0, w_1 = correct_weights(x_0, y_0, w)

w_0, w_1

w = np.array([5.02,1])

# +
plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
#x_1 = np.linspace(df['LSTAT'].min(), df["LSTAT"].max())
x_1 = df["LSTAT"]
x_b = np.zeros_like(x_1)
x = np.vstack([x_1,x_b]).T

difference = 0

for i in range(len(x_1)):
    x_n, y_n = df['LSTAT'][i], df['MEDV'][i]
    alpha = 0.01
    difference += alpha*np.array(correct_weights(x_n, y_n, w))
    change = difference/len(x_1)

w -= change
print(change)

y_pred = neuron2(x, w)
error = (df["MEDV"].values - y_pred)
mse = np.mean(error**2)
plt.title(mse)
plt.plot(x_1,y_pred, c='r')
plt.scatter(x_n, y_n, color="black")
# -



w = np.array([0.22925899, 0.12375   ])

# +
plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
#x_1 = np.linspace(df['LSTAT'].min(), df["LSTAT"].max())
x_1 = df["LSTAT"]
x_b = np.zeros_like(x_1)
x = np.vstack([x_1,x_b]).T

y_pred = neuron2(x, w)
y_actual = df["MEDV"]
mse_val = mse(y_actual, y_pred)
plt.title(mse_val)
plt.plot(x_1,y_pred, c='r')
plt.scatter(x_n, y_n, color="black")


# -

def mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted)**2)


def l_mse(X,y, w):
    y_pred = neuron2(X,w)
    return mse(y, y_pre


# +
w2 = np.array([0.22925899, 1.12375   ])
w3 = np.array([3.22925899, 0.12375   ])


w2_loss = l_mse(X_train, y_train, w2)
print(f"Model 1 (w=[-0.6, 30]) Loss: {w2_loss:.2f}")

w3_loss = l_mse(X_train, y_train, w3)
print(f"Model 2 (w=[-0.8, 30]) Loss: {w3_loss:.2f}")


# -

def mse_gradient(X,y,w):
    y_pred = neuron2(X,w)
    gradient = -(2/len(y)) * X.T @ (y - y_pred)
    return gradient


mse_gradient(X_train, y_train, w2)

mse_gradient(X_train, y_train, w3)



# +
plt.figure(figsize=(5, 3))
w = np.array([4.22925899, 6.12375   ])
plt.scatter(df['LSTAT'], df['MEDV'])
#x_1 = np.linspace(df['LSTAT'].min(), df["LSTAT"].max())
X_train = np.column_stack([df['LSTAT'], np.ones(len(df))])
y_train = df['MEDV'].values

iterations = 140
learning_rate = 0.0001
for _ in range(iterations):
    gradient = mse_gradient(X_train, y_train, w)
    loss = l_mse(X_train, y_train, w)
    w = w - learning_rate * grad
    print(w, loss)
    
    
y_pred = neuron2(x, w)
y_actual = df["MEDV"]
mse_val = mse(y_actual, y_pred)
plt.title(mse_val)
plt.plot(x_1,y_pred, c='r')
plt.scatter(x_n, y_n, color="black")
# -


