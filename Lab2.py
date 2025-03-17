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


def neuron2(x,w):
    return x@w


# +
# optional !pip install --user sklearn
# -

from sklearn.datasets import fetch_openml
import pandas as pd

boston = fetch_openml(name="boston", version=1, as_frame=True)

df = boston.frame

df

plt.figure(figsize=(10, 6))
plt.scatter(df['LSTAT'], df['MEDV'])
x = ...
w = ...
y_pred = neuron2(x, w)
plt.plot(x,y_pred)


